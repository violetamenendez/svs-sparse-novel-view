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

import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--configdir", type=str, default='./configs/',
                        help='dataset config files with splits')
    parser.add_argument('--with_rgb_loss', action='store_true')
    parser.add_argument('--imgScale_train', type=float, default=1.0)
    parser.add_argument('--imgScale_test', type=float, default=1.0)
    parser.add_argument('--img_downscale', type=float, default=1.0)
    parser.add_argument('--pad', type=int, default=24)

    # loader options
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--pts_dim", type=int, default=3)
    parser.add_argument("--dir_dim", type=int, default=3)
    parser.add_argument("--alpha_feat_dim", type=int, default=8)
    parser.add_argument('--net_type', type=str, default='v0')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['dtu', 'blender', 'llff', 'dtu_ft'])
    parser.add_argument('--use_color_volume', default=False, action="store_true",
                        help='project colors into a volume without indexing from image everytime')
    parser.add_argument('--use_density_volume', default=False, action="store_true",
                        help='point sampling with density')

    # training options
    parser.add_argument("--netdepth", type=int, default=6,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=6,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128,
                        help='channels per layer in fine network')

    parser.add_argument("--chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--precision", type=int, default=32, choices=[16,32],
                        help='select 32 bits precision or mixed precision')
    parser.add_argument("--acc_grad", type=int, default=1,
                        help='number of batches to accumulate gradients over')

    # Hyperparameters
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_disc", type=float, default=1e-4,
                        help='learning rate for the discriminator')
    parser.add_argument('--decay_step', nargs='+', type=int, default=[5000, 8000, 9000],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay amount')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    parser.add_argument('--lambda_rec', type=int, default=200,
                        help='Reconstruction loss weight coefficient')
    parser.add_argument('--lambda_depth_reg', type=float, default=0.1,
                        help='Depth regularisation coefficient')
    parser.add_argument('--lambda_depth_smooth', type=float, default=0.1,
                        help='Depth smoothness loss coefficient')
    parser.add_argument('--lambda_distortion', type=float, default=0.1,
                        help='Depth smoothness loss coefficient')
    parser.add_argument('--lambda_perc', type=float, default=0.1,
                        help='Depth smoothness loss coefficient')
    parser.add_argument('--lambda_adv', type=float, default=0.5,
                        help='Depth smoothness loss coefficient')

    # Losses
    parser.add_argument("--gan_loss", type=str, default=None, choices=["naive", "lsgan"],
                        help='type of gan loss to use during training')
    parser.add_argument("--gan_type", type=str, default=None, choices=["basic", "n_layers", "pixel", "graf"],
                        help='type of discriminator architecture')
    parser.add_argument("--getIntermFeat", action='store_true',
                        help='enforce feature matching of discriminator layers')
    parser.add_argument('--with_depth_loss', action='store_true',
                        help='enforce depth supervision')
    parser.add_argument('--with_depth_loss_rec', action='store_true',
                        help='enforce depth reconstruction')
    parser.add_argument('--with_depth_loss_reg', action='store_true',
                        help='enforce depth regularisation')
    parser.add_argument('--with_depth_smoothness', action='store_true',
                        help='enforce edge-aware depth smoothness')
    parser.add_argument('--with_distortion_loss', action='store_true',
                        help='enforce ray distortion loss')
    parser.add_argument('--with_perceptual_loss', action='store_true',
                        help='enforce ray distortion loss')
    parser.add_argument("--depth_path", type=str, default=None,
                        help='GT depth for adversarial loss')
    parser.add_argument("--finetune_scene", type=str, default=None,
                        help='name of scene to fine-tune model')
    parser.add_argument("--seed_everything", type=int, default=-1,
                        help='Set the random seed. Positive numbers only')
    parser.add_argument('--use_closest_views', action='store_true',
                        help='train using the closest views to the target image')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=128,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--scale_anneal", type=float, default=0.0025,
                        help='patch scale reduction over training')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--pts_embedder", action='store_true',
                        help='enable positional encoding  for points')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')



    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')


    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=20,
                        help='frequency of visualize the depth')
    parser.add_argument("--save_dir", type=str, default="runs",
                        help='path to directory to save result, logs and ckpts')

    # test options
    parser.add_argument("--vis_cnn", action='store_true',
                        help='activate CNN visualisation tests')
    parser.add_argument("--save_test", type=str, default="test_suite",
                        help='path to directory to save test results')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()