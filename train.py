# Copyright 2022 BBC and University of Surrey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imageio
import logging, coloredlogs
from pathlib import Path

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

# pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from collections import defaultdict

# models
from networks import Embedding, MVSNeRF, MVSNet, MVSNeRF_G, BasicDiscriminator, NLayerDiscriminator, PixelDiscriminator, GRAFDiscriminator

# metrics
from kornia.metrics import psnr, ssim
import lpips

from utils import build_rays, visualize_depth
from renderer import rendering
from opt import config_parser
from data import dataset_dict
from losses import total_variation_loss, get_disparity_smoothness, distortion_loss

logging.captureWarnings(True)
coloredlogs.install(
   level=logging.WARNING,
   fmt="%(asctime)s %(name)s:%(module)s.%(funcName)s[%(lineno)d] %(levelname)s %(message)s",
   datefmt="%F %T"
)

class MVSNeRFSystem(LightningModule):
    def __init__(self, hparams, pts_embedder=True, use_mvs=True, dir_embedder=False):
        # For MVSSystem we need to call it with use_mvs=True, dir_embedder=False, pts_embedder=True
        super(MVSNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        # Save system config
        self.hparams.pts_embedder = pts_embedder
        self.hparams.dir_embedder = dir_embedder
        self.hparams.use_mvs = use_mvs

        # From MVSSystem
        self.hparams.feat_dim = 8+3*4
        self.idx = 0 # validation step counter

        # Losses
        #self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.L1Loss()
        self.criterionFeat = nn.L1Loss()
        self.tv_loss = total_variation_loss
        self.depth_smooth = get_disparity_smoothness
        self.dist_loss = distortion_loss
        self.perc_loss = lpips.LPIPS(net='alex')

        # Metrics
        self.lpips = self.perc_loss

        # Define embedders
        self.embedding_xyz = Embedding(self.hparams.pts_dim, self.hparams.multires) if pts_embedder else None
        self.embedding_dir = Embedding(self.hparams.dir_dim, self.hparams.multires_views) if dir_embedder else None
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        # Define coarse NeRF
        self.input_ch = self.embedding_xyz.out_channels if self.embedding_xyz else self.hparams.pts_dim
        self.input_ch_views = self.embedding_dir.out_channels if self.embedding_dir else self.hparams.dir_dim
        self.output_ch = 4
        skips = [4] # Maybe this should be defined somewhere else
        self.nerf_coarse = MVSNeRF(D=self.hparams.netdepth, W=self.hparams.netwidth,
                 input_ch_pts=self.input_ch, output_ch=self.output_ch, skips=skips,
                 input_ch_views=self.input_ch_views, input_ch_feat=self.hparams.feat_dim, net_type=self.hparams.net_type)
        self.models = [self.nerf_coarse]

        # Default is no fine network.
        if self.hparams.N_importance > 0:
            self.nerf_fine = MVSNeRF(D=self.hparams.netdepth, W=self.hparams.netwidth,
                 input_ch_pts=self.input_ch, output_ch=self.output_ch, skips=skips,
                 input_ch_views=self.input_ch_views, input_ch_feat=self.hparams.feat_dim)
            self.models += [self.nerf_fine]

        # Encoding volume (creates cost volume and then enc vol)
        if use_mvs:
            self.encoding_net = MVSNet()
            self.models += [self.encoding_net]

        # Define Generator: Encoding volume -> NeRF -> Vol Rendering
        self.generator = MVSNeRF_G(self.hparams,
                                   self.nerf_coarse, self.encoding_net,
                                   self.embedding_xyz, self.embedding_dir)

        # Define type of GAN
        if self.hparams.gan_loss == "naive":
            self.adversarial_loss = nn.BCELoss(reduction='mean')
        elif self.hparams.gan_loss == "lsgan":
            self.adversarial_loss = nn.MSELoss(reduction='mean')

        # Define Discriminator architecture
        if self.hparams.gan_type == 'basic':       # default
            self.discriminator = BasicDiscriminator(torch.tensor([1, self.hparams.patch_size * self.hparams.patch_size, 3]), gan_type=self.hparams.gan_loss)
        elif self.hparams.gan_type == 'n_layers':  # PatchGAN classifier (https://github.com/znxlwm/pytorch-pix2pix)
            self.discriminator = NLayerDiscriminator(self.hparams.patch_size, 3, 64, 3, getIntermFeat=self.hparams.getIntermFeat)
        elif self.hparams.gan_type == 'pixel':     # PixelGAN classify if each pixel is real or fake (https://github.com/znxlwm/pytorch-pix2pix)
            self.discriminator = PixelDiscriminator(self.hparams.patch_size, 3, 64)
        elif self.hparams.gan_type == 'graf':      # patch annealing from GRAF (https://github.com/autonomousvision/graf)
            self.discriminator = GRAFDiscriminator(imsize=self.hparams.patch_size, nc=3, ndf=64)

        # Add adversarial depth supervision - Default off
        self.with_depth_loss = self.hparams.with_depth_loss
        if self.with_depth_loss:
            self.depth_disc = NLayerDiscriminator(self.hparams.patch_size, 1, 64, 3)
        #####################################

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]

        kwargs = {}
        if self.hparams.finetune_scene != None:
            kwargs['scene'] = self.hparams.finetune_scene

        # Training dataset
        if self.hparams.dataset_name == 'dtu':
            kwargs['max_len'] = -1
        if self.hparams.dataset_name == 'llff':
            kwargs['depth_path'] = self.hparams.depth_path
        self.train_dataset = dataset(self.hparams.datadir,
                                     split='train',
                                     config_dir=self.hparams.configdir,
                                     downSample=self.hparams.imgScale_train,
                                     closest_views=self.hparams.use_closest_views,
                                     **kwargs)

        # Validation dataset
        if self.hparams.dataset_name == 'dtu':
            kwargs['max_len'] = 10
        if self.hparams.dataset_name == 'llff':
            kwargs['depth_path'] = None
        self.val_dataset = dataset(self.hparams.datadir,
                                   split='val',
                                   config_dir=self.hparams.configdir,
                                   downSample=self.hparams.imgScale_test,
                                   closest_views=self.hparams.use_closest_views,
                                   **kwargs)

        # Testing dataset
        if self.hparams.dataset_name == 'dtu':
            kwargs['max_len'] = -1
            # kwargs['img_wh'] = (960,640)
        if self.hparams.dataset_name == 'llff':
            kwargs['depth_path'] = None
        self.test_dataset = dataset(self.hparams.datadir,
                                    split='test',
                                    config_dir=self.hparams.configdir,
                                    downSample=self.hparams.imgScale_test,
                                    closest_views=self.hparams.use_closest_views,
                                    **kwargs)
        #####################################

    def configure_optimizers(self):
        eps = 1e-7
        optimizers, schedulers = [], []

        # Generator optimiser
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lrate, betas=(0.9, 0.999))
        scheduler_G = CosineAnnealingLR(self.opt_G, T_max=self.hparams.num_epochs, eta_min=eps)
        optimizers.append(self.opt_G)
        schedulers.append(scheduler_G)

        # Discriminator optimiser
        if self.hparams.gan_type != None:
            self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lrate_disc, betas=(0.9, 0.999))
            scheduler_D = CosineAnnealingLR(self.opt_D, T_max=self.hparams.num_epochs, eta_min=eps)
            optimizers.append(self.opt_D)
            schedulers.append(scheduler_D)

        # Depth discriminator optimiser - default off
        if self.with_depth_loss:
            self.opt_depth_D = torch.optim.Adam(self.depth_disc.parameters(), lr=self.hparams.lrate_disc, betas=(0.9, 0.999))
            scheduler_depth_d = CosineAnnealingLR(self.opt_depth_D, T_max=self.hparams.num_epochs, eta_min=eps)
            optimizers.append(self.opt_depth_D)
            schedulers.append(scheduler_depth_d)

        return optimizers, schedulers

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0,
                          batch_size=1, #self.hparams.batch_size,
                          # NOTE: batch refers to number of images,
                          # but NeRF in practice takes ray batches
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        """
        Unnormalise images for visualisation
        Using ImageNet mean and std
        shape == N V C H W
        """
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def forward(self, data_mvs):
        rgb, target_s, depth_pred, depth, weights, t_vals = self.generator(data_mvs, self.global_step)

        return {'rgb': rgb,
                'target_s': target_s,
                'depth_pred': depth_pred,
                'depth_gt': depth,
                'weights': weights,
                't_vals': t_vals}

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        logging.info("TRAINING STEP")
        results = self(batch)

        # Depth total variation regularisation - not on paper
        tv_depth_loss = 0
        if self.hparams.with_depth_loss_reg:
            depth_patch = results['depth_pred'].reshape(-1, self.hparams.patch_size, self.hparams.patch_size, 1).squeeze(-1)
            tv_depth_loss = self.hparams.lambda_depth_reg * self.tv_loss(depth_patch)
            self.log('tv_depth_loss', tv_depth_loss)

        # Depth smoothness - equation 12
        depth_smooth_loss = 0
        if self.hparams.with_depth_smoothness:
            depth_patch = results['depth_pred'].reshape(-1, self.hparams.patch_size, self.hparams.patch_size, 1)
            img_patch = results['rgb'].reshape(-1, self.hparams.patch_size, self.hparams.patch_size, 3)
            depth_smooth_loss = self.hparams.lambda_depth_smooth * self.depth_smooth(depth_patch, img_patch)
            self.log('depth_smooth_loss', depth_smooth_loss)

        # Distortion loss - equation 13
        # Regularisation for interval-based models
        distortion_loss = 0
        if self.hparams.with_distortion_loss:
            distortion_loss = self.hparams.lambda_distortion * self.dist_loss(results['weights'], results['t_vals'])
            self.log('distortion_loss', distortion_loss)

        # Perceptual loss - LPIPS
        perceptual_loss = 0
        if self.hparams.with_perceptual_loss:
            pred_patch = results['rgb'].reshape(-1, self.hparams.patch_size, self.hparams.patch_size, 3).permute(0,3,1,2).float() * 2 - 1.0
            gt_patch = results['target_s'].reshape(-1, self.hparams.patch_size, self.hparams.patch_size, 3).permute(0,3,1,2).float() * 2 - 1.0
            perceptual_loss = self.hparams.lambda_perc * self.perc_loss(pred_patch, gt_patch)
            self.log('perceptual_loss', perceptual_loss)

        if self.hparams.gan_type != None:
            # Adversarial training

            # Train Generator
            if optimizer_idx == 0:
                # Generator loss
                G_pred_fake = self.discriminator(results['rgb'])
                if self.hparams.getIntermFeat:
                    D_interfeat_fake = G_pred_fake[:-1]
                    G_pred_fake = G_pred_fake[-1]

                G_fake_loss = self.adversarial_loss(G_pred_fake, torch.ones_like(G_pred_fake))
                G_fake_loss = self.hparams.lambda_adv * G_fake_loss
                logging.info("Generator loss: Disc prediction "+str(G_pred_fake)+", Gen loss "+str(G_fake_loss))
                self.log('G_fake_loss', G_fake_loss)

                # Intermediate feature matching - default off
                G_feat_loss = 0
                if self.hparams.getIntermFeat:
                    real_img = results['target_s'].detach()
                    D_pred_real = self.discriminator(real_img)
                    D_interfeat_real = D_pred_real[:-1]

                    assert len(D_interfeat_fake) == len(D_interfeat_real), \
                        f"Fake feat maps {len(D_interfeat_fake)}, Real feat maps {len(D_interfeat_real)}"
                    for feat_fake, feat_real in zip(D_interfeat_fake, D_interfeat_real):
                        G_feat_loss += self.criterionFeat(feat_fake, feat_real)
                    self.log('G_feat_loss', G_feat_loss)

                # Adversarial depth supervision - default off
                G_depth_fake_loss = 0
                if self.with_depth_loss:
                    G_depth_pred_fake = self.depth_disc(results['depth_pred'])
                    # depth_pred.reshape(-1, self.args.patch_size, self.args.patch_size)
                    G_depth_fake_loss = self.adversarial_loss(G_depth_pred_fake, torch.ones_like(G_depth_pred_fake))
                    self.log('G_depth_fake_loss', G_depth_fake_loss)

                # Depth reconstruction - default off
                rec_depth_loss = 0
                if self.hparams.with_depth_loss_rec:
                    rec_depth_loss = self.loss(results['depth_pred'], results['depth_gt'])
                    self.log('rec_depth_loss', rec_depth_loss)

                # Reconstruction loss - equation 9
                G_rec_loss = self.hparams.lambda_rec * self.loss(results['rgb'], results['target_s'])
                self.log('G_rec_loss', G_rec_loss)

                total_loss = G_fake_loss \
                    + G_feat_loss \
                    + G_depth_fake_loss + rec_depth_loss \
                    + G_rec_loss \
                    + tv_depth_loss \
                    + depth_smooth_loss \
                    + distortion_loss \
                    + perceptual_loss
                self.log('G_loss', total_loss)

            # Train Discriminator
            if optimizer_idx == 1:
                # Fake detection and loss
                fake_img = results['rgb'].detach()
                D_pred_fake = self.discriminator(fake_img)
                if self.hparams.getIntermFeat:
                    D_pred_fake = D_pred_fake[-1]

                D_fake_loss = self.adversarial_loss(D_pred_fake, torch.zeros_like(D_pred_fake))
                logging.info("Discriminator fake loss: Disc prediction "+str(D_pred_fake)+", Disc loss "+str(D_fake_loss))
                self.log('D_fake_loss', D_fake_loss)

                # Real detection and loss
                real_img = results['target_s'].detach()
                D_pred_real = self.discriminator(real_img)
                if self.hparams.getIntermFeat:
                    D_pred_real = D_pred_real[-1]

                D_real_loss = self.adversarial_loss(D_pred_real, torch.ones_like(D_pred_real))
                logging.info("Discriminator real loss: Disc prediction "+str(D_pred_real)+", Disc loss "+str(D_real_loss))
                self.log('D_real_loss', D_real_loss)

                total_loss = (D_fake_loss + D_real_loss) / 2

            # Train Depth Discriminator
            if optimizer_idx == 2:
                # Fake depth supervision
                D_depth_pred_fake = self.depth_disc(results['depth_pred'].detach())
                # depth_pred.reshape(-1, self.args.patch_size, self.args.patch_size)
                D_depth_fake_loss = self.adversarial_loss(D_depth_pred_fake, torch.zeros_like(D_depth_pred_fake))

                logging.info("Discriminator fake loss: Disc prediction "+str(D_depth_fake_loss)+", Disc loss "+str(D_depth_fake_loss))
                self.log('D_depth_fake_loss', D_depth_fake_loss)

                # Real depth supervision
                D_depth_pred_real = self.depth_disc(results['depth_gt'].detach())
                # depth_pred.reshape(-1, self.args.patch_size, self.args.patch_size)
                D_depth_real_loss = self.adversarial_loss(D_depth_pred_real, torch.ones_like(D_depth_pred_real))

                logging.info("Depth discriminator real loss: Disc prediction "+str(D_depth_pred_real)+", Disc loss "+str(D_depth_real_loss))
                self.log('D_depth_real_loss', D_depth_real_loss)

                total_loss = (D_depth_fake_loss + D_depth_real_loss) / 2

                self.log('D_depth_loss', total_loss)
        else:
            # Normal training
            total_loss = self.loss(results['rgb'], results['target_s']) \
                       + self.hparams.lambda_depth_reg * tv_depth_loss \
                       + self.hparams.lambda_depth_smooth * depth_smooth_loss \
                       + self.hparams.lambda_distortion * distortion_loss

        self.log('train_loss', total_loss, prog_bar=True)

        # Metrics
        with torch.no_grad():
            psnr_ = psnr(results['rgb'], results['target_s'], 1)
            self.log('train_PSNR', psnr_, prog_bar=True)


        logging.info("Training metrics: LOSS "+str(total_loss)+", PSNR "+str(psnr_))

        return {'loss': total_loss}

    def validation_step(self, batch, batch_nb):
        logging.info("VALIDATION STEP")
        # TODO - Old model code. Should refactor to match training step

        imgs, proj_mats = batch['images'], batch['proj_mats']
        near_fars = batch['near_fars']
        depths = batch['depths_h']

        self.encoding_net.train()
        N, V, C, H, W = imgs.shape
        logging.info("image batch: imgs "+str(imgs.shape))


        ##################  rendering #####################
        with torch.no_grad():

            self.hparams.img_downscale = torch.rand((1,)) * 0.75 + 0.25  # for super resolution
            w2cs = batch['w2cs']
            c2ws, intrinsics = batch['c2ws'], batch['intrinsics']

            # Encoding volume
            volume_feature, img_feat, _ = self.encoding_net(imgs[:, :3],
                                                      proj_mats[:, :3],
                                                      near_fars[0,0],
                                                      pad=self.hparams.pad)
            logging.info("volume_feature "+str(volume_feature.shape))

            imgs = self.unpreprocess(imgs)
            rgbs, depth_preds = [],[]
            for chunk_idx in range(H*W//self.hparams.chunk + int(H*W%self.hparams.chunk>0)):

                rays_pts, rays_dir, _, rays_NDC, depth_candidates, depth, t_vals = \
                    build_rays(imgs, depths, w2cs, c2ws, intrinsics, near_fars,
                               self.hparams.N_samples, stratified=False,
                               pad=self.hparams.pad, chunk=self.hparams.chunk,
                               idx=chunk_idx, val=True, isRandom=False)

                rgb, disp, acc, depth_pred, density_ray = rendering(self.hparams, batch,
                                                                    rays_pts, rays_NDC,
                                                                    depth_candidates,
                                                                    rays_dir,
                                                                    volume_feature,
                                                                    imgs[:, :-1],
                                                                    img_feat=None,
                                                                    network_fn=self.nerf_coarse,
                                                                    embedding_pts=self.embedding_xyz,
                                                                    embedding_dir=self.embedding_dir,
                                                                    white_bkgd=self.hparams.white_bkgd)
                rgbs.append(rgb.squeeze(0));depth_preds.append(depth_pred.squeeze(0))
                logging.info("render outs rgb depth "+str(rgb.shape)+", "+str(depth_pred.shape))

            # Validation processes only one image at a time, so batch N==1
            imgs = imgs[0] # [1,V,3,H,W] -> [V,3,H,W]
            tgt_img = imgs[-1].unsqueeze(0) # [1,3,H,W] Target view is the last view
            rgb, depth_r = torch.clamp(torch.cat(rgbs).reshape(1, H, W, 3).permute(0,3,1,2),0,1), torch.cat(depth_preds).reshape(H, W)
            img_err_abs = (rgb - tgt_img).abs()
            logging.info("rgb, depth_r, img_err_abs "+str(rgb.shape)+", "+str(depth_r.shape)+", "+str(img_err_abs.shape))


            log = {'val_loss': self.loss(rgb, tgt_img)}
            psnr_ = psnr(rgb, tgt_img, 1)
            ssim_ = ssim(rgb, tgt_img, 5).mean()
            lpips_ = self.lpips(rgb, tgt_img)
            log['val_psnr'] = psnr_
            log['val_ssim'] = ssim_
            log['val_lpips'] = lpips_
            logging.info("Validation metrics: PSNR "+str(psnr_)+", SSIM "+str(ssim_)+", LPIPS "+str(lpips_))


            minmax = [2.0, 6.0]
            depth_pred_r_, _ = visualize_depth(depth_r, minmax)
            self.logger.experiment.add_images('val/depth_gt_pred_err', depth_pred_r_[None], self.global_step)
            logging.info("Depth rays? "+str(depth_r.shape)+", "+str(depth_pred_r_.shape))

            img_vis = torch.cat((imgs.cpu(), rgb.cpu(), img_err_abs.cpu()*5), dim=0) # [V 3 H W]
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step) # only show one sample from the batch
            logging.info("img_vis "+str(img_vis.shape))

            img_vis = torch.cat((img_vis,depth_pred_r_[None]),dim=0)
            logging.info("img_vis "+str(img_vis.shape))

            img_vis = img_vis.permute(2,0,3,1).reshape(img_vis.shape[2],-1,3).numpy()
            logging.info("img_vis "+str(img_vis.shape))

            save_dir_vis = self.hparams.save_dir / self.hparams.expname / 'val_images'
            save_dir_vis.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(save_dir_vis / f'{self.global_step:08d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))
            self.idx += 1

        del rays_NDC, rays_dir, rays_pts, volume_feature

        return log

    def validation_epoch_end(self, outputs):
        logging.info("End of validation epoch")

        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        mean_lpips = torch.stack([x['val_lpips'] for x in outputs]).mean()

        self.log('val_loss', mean_loss, prog_bar=True)
        self.log('val_PSNR', mean_psnr, prog_bar=True)
        self.log('val_SSIM', mean_ssim, prog_bar=False)
        self.log('val_LPIPS', mean_lpips, prog_bar=False)

        return

    def test_step(self, batch, batch_nb):
        logging.info("TEST STEP")
        # TODO - refactor

        imgs, proj_mats = batch['images'], batch['proj_mats']
        near_fars = batch['near_fars']

        self.encoding_net.train()
        N, V, C, H, W = imgs.shape
        logging.info("image batch: imgs "+str(imgs.shape))

        with torch.no_grad():

            self.hparams.img_downscale = torch.rand((1,)) * 0.75 + 0.25
            w2cs = batch['w2cs']
            c2ws, intrinsics = batch['c2ws'], batch['intrinsics']
            depths = batch['depths_h']

            # Encoding volume
            volume_feature, img_feat, _ = self.encoding_net(imgs[:, :3],
                                                            proj_mats[:, :3],
                                                            near_fars[0,0],
                                                            pad=self.hparams.pad)
            logging.info("volume_feature "+str(volume_feature.shape))

            imgs = self.unpreprocess(imgs)
            rgbs, depth_preds = [],[]
            for chunk_idx in range(H*W//self.hparams.chunk + int(H*W%self.hparams.chunk>0)):

                rays_pts, rays_dir, _, rays_NDC, depth_candidates, depth, t_vals = \
                    build_rays(imgs, depths, w2cs, c2ws, intrinsics, near_fars, self.hparams.N_samples, stratified=False, pad=self.hparams.pad, chunk=self.hparams.chunk, idx=chunk_idx, val=True, isRandom=False)

                rgb, disp, acc, depth_pred, density_ray = rendering(self.hparams, batch,
                                                                    rays_pts, rays_NDC,
                                                                    depth_candidates,
                                                                    rays_dir,
                                                                    volume_feature,
                                                                    imgs[:, :-1],
                                                                    img_feat=None,
                                                                    network_fn=self.nerf_coarse,
                                                                    embedding_pts=self.embedding_xyz,
                                                                    embedding_dir=self.embedding_dir,
                                                                    white_bkgd=self.hparams.white_bkgd)
                rgbs.append(rgb.squeeze(0));depth_preds.append(depth_pred.squeeze(0))
                logging.info("render outs rgb depth "+str(rgb.shape)+", "+str(depth_pred.shape))

            # batch N==1
            imgs = imgs[0] # [V,3,H,W]
            tgt_img = imgs[-1].unsqueeze(0) # [1,3,H,W]
            rgb, depth_r = torch.clamp(torch.cat(rgbs).reshape(1, H, W, 3).permute(0,3,1,2),0,1), torch.cat(depth_preds).reshape(H, W)
            img_err_abs = (rgb - tgt_img).abs()
            logging.info("rgb, depth_r, img_err_abs "+str(rgb.shape)+", "+str(depth_r.shape)+", "+str(img_err_abs.shape))

            # Metrics
            log = {}
            psnr_ = psnr(rgb, tgt_img, 1)
            ssim_ = ssim(rgb, tgt_img, 5).mean()
            lpips_ = self.lpips(rgb, tgt_img)
            log['test_psnr'] = psnr_
            log['test_ssim'] = ssim_
            log['test_lpips'] = lpips_
            logging.info("Test metrics: PSNR "+str(psnr_)+", SSIM "+str(ssim_)+", LPIPS "+str(lpips_))


            minmax = [2.0, 6.0]
            depth_pred_r_, _ = visualize_depth(depth_r, minmax)
            self.logger.experiment.add_images('val/depth_gt_pred_err', depth_pred_r_[None], self.global_step)
            logging.info("Depth rays? "+str(depth_r.shape)+", "+str(depth_pred_r_.shape))

            img_vis = torch.cat((imgs.cpu(), rgb.cpu(), img_err_abs.cpu()*5), dim=0) # [V 3 H W]
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step) # only show one sample from the batch
            logging.info("img_vis "+str(img_vis.shape))

            img_vis = torch.cat((img_vis,depth_pred_r_[None]),dim=0)
            logging.info("img_vis "+str(img_vis.shape))

            img_vis = img_vis.permute(2,0,3,1).reshape(img_vis.shape[2],-1,3).numpy()
            logging.info("img_vis "+str(img_vis.shape))

            save_dir_vis = self.hparams.save_dir / self.hparams.expname / 'val_images'
            save_dir_vis.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(save_dir_vis / f'{self.global_step:08d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))
            self.idx += 1

        del rays_NDC, rays_dir, rays_pts, volume_feature

        return log

    def test_epoch_end(self, outputs):
        logging.info("End of test epoch")

        mean_psnr = torch.stack([x['test_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['test_ssim'] for x in outputs]).mean()
        mean_lpips = torch.stack([x['test_lpips'] for x in outputs]).mean()

        self.log('test_PSNR', mean_psnr, prog_bar=True)
        self.log('test_SSIM', mean_ssim, prog_bar=False)
        self.log('test_LPIPS', mean_lpips, prog_bar=False)

        return

def main():
    torch.set_default_dtype(torch.float32)
    hparams = config_parser()

    if hparams.seed_everything >= 0:
        pl.seed_everything(hparams.seed_everything)

    hparams.save_dir = Path(hparams.save_dir)
    system = MVSNeRFSystem(hparams, pts_embedder=hparams.pts_embedder)

    save_dir_ckpts = hparams.save_dir / hparams.expname / 'ckpts'
    save_dir_ckpts.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir_ckpts,
                                          filename='{epoch:02d}-{step}-{val_loss:.3f}',
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=True,
                                          save_top_k=5,
                                          save_last=True)

    logger = loggers.TestTubeLogger(
        save_dir=hparams.save_dir,
        name=hparams.expname,
        debug=False,
        create_git_tag=False
    )

    # Load checkpoints from given path or resume from existing
    resume_ckpt = None
    if hparams.ckpt:
        resume_ckpt = hparams.ckpt
    elif Path(save_dir_ckpts / 'last.ckpt').exists():
        resume_ckpt = save_dir_ckpts / 'last.ckpt'

    hparams.num_gpus = 1
    trainer = Trainer(resume_from_checkpoint=resume_ckpt,
                      max_epochs=hparams.num_epochs,
                      checkpoint_callback=True,
                      callbacks=checkpoint_callback,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch = max(system.hparams.num_epochs//system.hparams.N_vis,1),
                      benchmark=True,
                      precision=system.hparams.precision,
                      accumulate_grad_batches=hparams.acc_grad)

    trainer.fit(system)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # python train_mvs_nerf_pl.py --expname dtu_example_lala --num_epochs 6
    # --use_viewdirs --dataset_name dtu --datadir dtu

    try:
        main()
    finally:
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary(abbreviated=True))
