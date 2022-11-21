import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from inplace_abn import InPlaceABN

from utils import homo_warp, build_rays
from renderer import rendering

##############################  NeRF Net models  ##############################
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        NeRf Positional Encoder

        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

class Renderer(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        NeRF
        """
        super(Renderer, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts = input_ch
        self.in_ch_views = input_ch_views
        self.in_ch_feat = input_ch_feat

        # Points encoding layers
        self.pts_linears = nn.ModuleList()
        for i in range(D-1):
            if i == 0:
                self.pts_linears.append(nn.Linear(self.in_ch_pts, W, bias=True))
            if i in skips:
                self.pts_linears.append(nn.Linear(W + self.in_ch_pts, W))
            else:
                self.pts_linears.append(nn.Linear(W, W, bias=True))

        # We want to bias the 3D points by the per-voxel neural
        # features (interpolated from the encoding volume)
        self.pts_bias = nn.Linear(self.in_ch_feat, W)

        if use_viewdirs:
            # Direction encoding layers
            self.views_linears = nn.ModuleList([nn.Linear(W + self.in_ch_views, W//2)])
            self.feature_linear = nn.Linear(W, W)# Final encoding layer
            self.alpha_linear = nn.Linear(W, 1)  # A
            self.rgb_linear = nn.Linear(W//2, 3) # RGB
        else:
            self.output_linear = nn.Linear(W, output_ch) # RGBA (output_ch = 4)

        # He initialisation - from a normal distribution
        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward_alpha(self, x):

        input_pts, input_feats = torch.split(x, [self.in_ch_pts, self.in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = torch.relu(self.alpha_linear(h))
        return alpha


    def forward(self, x):
        """
        Encodes input points+2Dfeatures+dir to rgb+sigma
        """
        logging.info("STEP FORWARD")

        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts,
                                                              self.in_ch_feat,
                                                              self.in_ch_views],
                                                          dim=-1)

        logging.info("inputs "+","+str(input_pts.shape)+"," \
            +str(input_feats.shape)+","+str(input_views.shape))

        # Encode inputs
        pts = input_pts
        bias = self.pts_bias(input_feats)

        for i, layer in enumerate(self.pts_linears):
            pts = layer(pts) * bias
            pts = F.relu(pts)
            if i in self.skips:
                pts = torch.cat([input_pts, pts], -1)

        if self.use_viewdirs:
            # Alpha depends only on 3D points
            alpha = torch.relu(self.alpha_linear(pts))

            # Concatenate point features and viewing direction (to get colour)
            feature = self.feature_linear(pts)
            pts = torch.cat([feature, input_views], -1)

            # Encode viewing direction
            for layer in self.views_linears:
                pts = layer(pts)
                pts = F.relu(pts)

            # Colour depends on point features and viewing direction
            rgb = torch.sigmoid(self.rgb_linear(pts))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(pts) # RGBA

        logging.info("outputs "+str(outputs.shape))
        return outputs

class Renderer_linear(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        NeRF where feature bias is linear
        """
        super(Renderer_linear, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts = input_ch
        self.in_ch_views = input_ch_views
        self.in_ch_feat = input_ch_feat

        # Points encoding layers
        self.pts_linears = nn.ModuleList()
        for i in range(D-1):
            if i == 0:
                self.pts_linears.append(nn.Linear(self.in_ch_pts, W, bias=True))
            if i in skips:
                self.pts_linears.append(nn.Linear(W + self.in_ch_pts, W))
            else:
                self.pts_linears.append(nn.Linear(W, W, bias=True))

        # We want to bias the 3D points by the per-voxel neural
        # features (interpolated from the encoding volume)
        self.pts_bias = nn.Linear(self.in_ch_feat, W)

        if use_viewdirs:
            # Direction encoding layers
            self.views_linears = nn.ModuleList([nn.Linear(W + self.in_ch_views, W//2)])
            self.feature_linear = nn.Linear(W, W)# Final encoding layer
            self.alpha_linear = nn.Linear(W, 1)  # A
            self.rgb_linear = nn.Linear(W//2, 3) # RGB
        else:
            self.output_linear = nn.Linear(W, output_ch) # RGBA (output_ch = 4)

        # He initialisation - from a normal distribution
        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward_alpha(self, x):

        input_pts, input_feats = torch.split(x, [self.in_ch_pts, self.in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) + bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha


    def forward(self, x):
        """
        Encodes input points+2Dfeatures+dir to rgb+sigma
        """

        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, self.in_ch_feat, self.in_ch_views], dim=-1)

        # Encode inputs
        pts = input_pts
        bias = self.pts_bias(input_feats)
        for i, layer in enumerate(self.pts_linears):
            pts = layer(pts) + bias
            pts = F.relu(pts)
            if i in self.skips:
                pts = torch.cat([input_pts, pts], -1)


        if self.use_viewdirs:
            # Alpha depends only on 3D points
            alpha = torch.relu(self.alpha_linear(pts))

            # Concatenate point features and viewing direction (to get colour)
            feature = self.feature_linear(pts)
            pts = torch.cat([feature, input_views], -1)

            # Encode viewing direction
            for layer in self.views_linears:
                pts = layer(pts)
                pts = F.relu(pts)

            # Colour depends on point features and viewing direction
            rgb = torch.sigmoid(self.rgb_linear(pts))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(pts) # RGBA

        return outputs

class MVSNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch_pts=3, output_ch=4, input_ch_views=3, input_ch_feat=8, skips=[4], net_type='v2'):
        """
        Wrapper around NeRF to select correct network type (either linear or not)
        """
        super(MVSNeRF, self).__init__()
        logging.info("Instantiating MVSNeRF network")
        logging.info("Network type"+str(net_type)+" with in_ch"+str(input_ch_pts) \
            +"out_ch"+str(output_ch)+"in_ch_views"+str(input_ch_views)+"in_ch_feat"+str(input_ch_feat))

        self.in_ch_pts, self.out_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch_pts, output_ch, input_ch_views, input_ch_feat

        if net_type == 'v0':
            # NeRF coarse
            # L_n = ReLU(FC(L_{n-1}) * neural_features)
            self.nerf = Renderer(D=D, W=W,input_ch_feat=input_ch_feat,
                        input_ch=input_ch_pts, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=True)
        elif net_type == 'v2':
            # NeRF fine
            # L_n = ReLU(FC(L_{n-1}) + neural_features)
            self.nerf = Renderer_linear(D=D, W=W,input_ch_feat=input_ch_feat,
                     input_ch=input_ch_pts, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)

    def forward_alpha(self, x):
        return self.nerf.forward_alpha(x)

    def forward(self, x):
        RGBA = self.nerf(x)
        return RGBA

class MVSNeRF_G(nn.Module):
    def __init__(self, args, nerf, encoding, embedding_pts, embedding_dir):
        """
        Generator using MVSNeRF
        """
        super(MVSNeRF_G, self).__init__()

        # Networks
        self.nerf = nerf
        self.encoding_net = encoding
        self.embedding_pts = embedding_pts
        self.embedding_dir = embedding_dir

        # Parameters
        self.N_rays = args.batch_size
        self.N_samples = args.N_samples
        self.args = args

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # Unnormalise images for visualisation
        # Using ImageNet mean and std
        # shape == N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def forward(self, x, step=0):
        imgs = x['images']
        proj_mats = x['proj_mats']
        near_fars = x['near_fars']
        w2cs = x['w2cs']
        c2ws = x['c2ws']
        intrinsics = x['intrinsics']
        depths = x['depths_h']

        # Neural Encoding Volume generation
        # imgs -> cost vol -> Enc vol (volume_feature)
        volume_feature, img_feat, depth_values = self.encoding_net(imgs[:, :3],
                                                                   proj_mats[:, :3],
                                                                   near_fars[0,0],
                                                                   pad=self.args.pad,
                                                                   vis_test=self.args.vis_cnn,
                                                                   test_dir=Path(self.args.save_test))
        imgs = self.unpreprocess(imgs) # unnormalise for visualisation

        # Ray generation from images and camera positions
        rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_depth_gt, t_vals = \
            build_rays(imgs, depths, w2cs, c2ws, intrinsics, near_fars, self.N_samples,
                       N_rays=self.N_rays, pad=self.args.pad,
                       patch_size=self.args.patch_size, scale_anneal=self.args.scale_anneal,
                       step=step, variable_patches=(self.args.gan_type=='graf'))

        # Render colours along rays from volume feature and images
        rgb, feats, weights, depth_pred, alpha = rendering(self.args, x,
                                                           rays_pts, rays_NDC,
                                                           depth_candidates,
                                                           rays_dir,
                                                           volume_feature,
                                                           imgs[:, :-1],
                                                           img_feat=None,
                                                           network_fn=self.nerf,
                                                           embedding_pts=self.embedding_pts,
                                                           embedding_dir=self.embedding_dir,
                                                           white_bkgd=self.args.white_bkgd)

        depth_pred = depth_pred.unsqueeze(-1)
        logging.info("render outs rgb targe "+str(rgb.shape)+", "+str(target_s.shape) \
            +", "+str(depth_pred.shape)+","+str(rays_depth_gt.shape))

        return rgb, target_s, depth_pred, rays_depth_gt, weights, t_vals

class BasicDiscriminator(nn.Module):
    def __init__(self, img_shape, gan_type = None):
        super().__init__()

        layers = [nn.Linear(int(torch.prod(img_shape)), 512),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.Linear(512, 256),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.Linear(256, 1)]

        if gan_type == None or gan_type == "naive":
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        img_flat = img.reshape(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, patch_size, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        """Construct a PatchGAN discriminator

        From https://github.com/znxlwm/pytorch-pix2pix

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.patch_size = patch_size
        self.getIntermFeat = getIntermFeat

        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
                            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                            nn.LeakyReLU(0.2, True)
                        ))
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw,
                              bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=kw,
                          stride=1,
                          padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))

        self.layers.append(nn.Sequential(nn.Conv2d(ndf * nf_mult,
                                                   1,
                                                   kernel_size=kw,
                                                   stride=1,
                                                   padding=padw)))  # output 1 channel prediction map

    def forward(self, img):
        # img.shape == [N,batch,ch]
        img = img.transpose(1,2) # [N, ch, batch]
        N, C, _ = img.shape
        img_patch = img.reshape((N, C, self.patch_size, self.patch_size)) # [1, 3, patch_h, patch_w]
        feat_maps = []
        x = img_patch
        for layer in self.layers:
            x = layer(x)
            feat_maps.append(x)

        if self.getIntermFeat:
            return feat_maps
        else:
            return feat_maps[-1]


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, patch_size, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        From https://github.com/znxlwm/pytorch-pix2pix

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        self.patch_size = patch_size

        use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, img):
        img_patch = img.reshape((1, self.patch_size, self.patch_size, 3))
        return self.net(img_patch)

class GRAFDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False):
        """Construct a GRAF discriminator

        From https://github.com/autonomousvision/graf

        Parameters:
            nc (int)     -- the number of channels in input images
            ndf (int)    -- the number of filters in the last conv layer
            imsize (int) -- side of square image
            hglip (bool) -- perform image flip
        """
        super(GRAFDiscriminator, self).__init__()
        self.nc = nc
        assert(imsize==32 or imsize==64 or imsize==128)
        self.imsize = imsize
        self.hflip = hflip

        SN = torch.nn.utils.spectral_norm
        IN = lambda x : nn.InstanceNorm2d(x)

        blocks = []
        if self.imsize==128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize==64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*2) x 16 x 16
            SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 4),
            IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            # nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

    def forward(self, input):
        input = input[:, :, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)  # (BxN_samples)xC -> BxCxHxW

        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input),1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        return self.main(input)

###############################  MVS Net models  ##############################
### From https://github.com/apchenstu/mvsnerf
###############################################################################

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

###############################  feature net  ################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        activ_maps = []
        x = self.conv0(x) # (B, 8, H, W)
        activ_maps.append(x)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        activ_maps.append(x)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        activ_maps.append(x)
        x = self.toplayer(x) # (B, 32, H//4, W//4)
        activ_maps.append(x)

        return x, activ_maps

class CostRegNet(nn.Module):
    """ 3D convolution
        Cost volume -> Neural Encoding Volume
    """

    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        activ_maps = []
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        activ_maps.append(conv0)
        activ_maps.append(conv2)
        activ_maps.append(conv4)

        x = self.conv6(self.conv5(conv4))
        activ_maps.append(x)
        x = conv4 + self.conv7(x)
        activ_maps.append(x)
        del conv4
        x = conv2 + self.conv9(x)
        activ_maps.append(x)
        del conv2
        x = conv0 + self.conv11(x)
        activ_maps.append(x)
        del conv0
        # x = self.conv12(x)
        return x, activ_maps

class MVSNet(nn.Module):
    def __init__(self,
                 num_groups=1,
                 norm_act=InPlaceABN,
                 levels=1):
        super(MVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [128,32,8]
        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = FeatureNet()

        self.chunk = 1024

        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        self.cost_reg_2 = CostRegNet(32+9, norm_act)

    def build_volume_cost(self, imgs, feats, proj_mats, depth_values, pad=0):
        """Build cost volume

        Args:
            imgs: (B, V, C, H, W)
            feats: (B, V, C, H, W)
            proj_mats: (B, V, 3, 4)
            depth_values: (B, D, H, W)
        """
        logging.info("BUILD VOL COST")
        logging.info("inputs "+str(imgs.shape)+","+str(feats.shape) \
        +","+str(proj_mats.shape)+","+str(depth_values.shape)+","+str(pad))

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        img_feat = torch.empty((B, 9 + 32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume # [B, C, D, h, w]
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        # Plane sweep volumes
        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs[1:], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:, (i + 1) * 3:(i + 2) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i + 1] = in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        logging.info("outputs "+str(img_feat.shape)+", "+str(in_mask.shape))
        return img_feat, in_masks

    def forward(self, imgs, proj_mats, near_far, pad=0,  return_color=False,
                lindisp=False, vis_test=False, test_dir=None):
        """Build encoding volume

        Cost volume -> Encoding volume
        Return:
            volume_feat: [B C D H W]
        """
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)
        logging.info("STEP FORWARD")
        logging.info("inputs "+str(imgs.shape)+","+str(proj_mats.shape) \
            +","+str(near_far.shape)+","+str(pad)+","+str(return_color)+","+str(lindisp))

        B, V, _, H, W = imgs.shape

        # 2D CNN - image feature maps
        imgs = imgs.reshape(B * V, 3, H, W)
        feats, activ_maps = self.feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)

        if vis_test:
            save_tensors_path = test_dir / f"2cnn_vis/tensors/"
            save_tensors_path.mkdir(parents=True, exist_ok=True)

            save_vis_path = test_dir / f"2cnn_vis/feat2viz/"
            save_vis_path.mkdir(parents=True, exist_ok=True)

            for id, amap in enumerate(activ_maps):

                # Save intermediate outputs
                torch.save(amap, save_tensors_path / f"activation_map_{id}.pt")

                # Vuisualise data
                # plot activ_maps[0]([3, 8, 512, 640])
                map_viz = feat2viz(activ_maps[id])
                # map_viz = activ_maps[id][:,0,:,:] # for flat images
                torchvision.utils.save_image(map_viz, save_vis_path / f"activation_map_{id}.png")
                # torchvision.utils.save_image(activ_maps[id][:,0,:,:],f"test_suite/not_trained/2cnn_vis/feat2viz/scan114_activ_map_{id}.png")

            # print(self.feature.conv0[0].conv.weight.shape)
            # print(self.feature.conv1[0].conv.weight.shape)
            # print(self.feature.conv2[0].conv.weight.shape)
            # print(self.feature.toplayer.weight.shape)


        imgs = imgs.view(B, V, 3, H, W)
        feats = feats.view(B, V, *feats.shape[1:])  # (B, V, C, h, w)

        D = 128
        t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        near, far = near_far  # assume batch size==1
        if not lindisp:
            depth_values = near * (1.-t_vals) + far * (t_vals)
        else:
            depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        depth_values = depth_values.unsqueeze(0)

        # Sweep Planes and Cost volume
        cost_vol, in_masks = self.build_volume_cost(imgs, feats, proj_mats, depth_values, pad=pad)
        if return_color:
            feats = torch.cat((cost_vol[:,:V*3].view(B, V, 3, *cost_vol.shape[2:]),in_masks.unsqueeze(2)),dim=2)

        if vis_test:
            save_cost_path = test_dir / f"cost_vol/tensors/"
            save_cost_path.mkdir(parents=True, exist_ok=True)
            torch.save(cost_vol, save_cost_path / f"cost_vol.pt")

        # 3D CNN - Neural Encoding Volume from Cost Volume
        volume_feat, activ_maps = self.cost_reg_2(cost_vol)  # (B, 1, D, h, w)
        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])
        if vis_test:
            save_tensors_path = test_dir / f"3cnn_vis/tensors/"
            save_tensors_path.mkdir(parents=True, exist_ok=True)

            save_vis_path = test_dir / f"3cnn_vis/feat2viz/layers"
            save_vis_path.mkdir(parents=True, exist_ok=True)

            for id, amap in enumerate(activ_maps):
                # Save layers
                torch.save(amap, save_tensors_path / f"activation_map_{id}.pt")

                # Visualise layers
                N, C, X, Y, Z = amap.shape
                for x in range(X):
                    map_viz = feat2viz(activ_maps[id][:,:,x,:,:])
                    torchvision.utils.save_image(map_viz, save_vis_path / f"activation_map_{id}_{x:03}.png")

            # print(self.feature.conv0[0].conv.weight.shape)
            # print(self.feature.conv1[0].conv.weight.shape)
            # print(self.feature.conv2[0].conv.weight.shape)
            # print(self.feature.toplayer.weight.shape)

        logging.info("outputs "+str(volume_feat.shape)+", "+str(feats.shape)+", "+str(depth_values.shape))
        return volume_feat, feats, depth_values

def feat2viz(feat: torch.Tensor) -> torch.Tensor:
    """Convert a dense feature map into a normalized PCA visualization.
    The PCA projection is computed using the features across the whole batch and normalized together.
    :param feat: (Tensor) (b, c, h, w) Input dense feature map.
    :return: (Tensor) (b, 3, h, w) Feature PCA visualization.
    """
    from sklearn.decomposition import PCA
    b, _, h, w = feat.shape
    feat = feat.permute(0, 2, 3, 1).flatten(0, 2).detach().cpu().numpy()  # (b, c, h, w) -> (b, h, w, c) -> (b*h*w, c)
    proj = PCA(n_components=3).fit_transform(feat)  # (n, c) -> (n, 3)
    proj -= proj.min(0)  # Normalize per channel
    proj /= proj.max(0)
    proj = torch.from_numpy(proj).reshape(b, h, w, 3).permute(0, 3, 1, 2)  # (b*h*w, 3) -> (b, h, w, 3) -> (b, 3, h, w)
    return proj
