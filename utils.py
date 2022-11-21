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

import re
import cv2
import numpy as np
import logging
from math import exp

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from kornia.utils import create_meshgrid

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: [N, H, W]
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

###########################  MVS  helper functions  ###########################
def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory



        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid

################################  Ray helpers  ################################
def patch_ray_sampler(patch_size, step, random_shift=True, random_scale=True,
                      min_scale=0.25, max_scale=1., scale_anneal=-1):

    w, h = torch.meshgrid([torch.linspace(-1,1,patch_size),
                           torch.linspace(-1,1,patch_size)])
    h = h.unsqueeze(2)
    w = w.unsqueeze(2)

    if scale_anneal>0:
        k_iter = step // 1000 * 3
        min_scale = max(min_scale, max_scale * exp(-k_iter*scale_anneal))
        min_scale = min(0.9, min_scale)
    else:
        min_scale = min_scale

    scale = 1
    if random_scale:
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
        h = h * scale
        w = w * scale

    if random_shift:
        max_offset = 1-scale.item()
        h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
        w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

        h += h_offset
        w += w_offset

    return torch.cat([h, w], dim=2)

def get_rays_mvs(H, W, intrinsic, c2w, N_rays=1024, isRandom=True, chunk=-1, idx=-1,
                 N_patches=None, patch_size=-1, scale_anneal=-1, step=0, variable_patches=False):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Args:
    - H,W: int
    - intrinsics: [N,3,3]
    - c2w: [N,4,4]

    Return:
        rays_o: [N, H*W, 3], the origin of the rays in world coordinate
        rays_d: [N, H*W, 3], the normalized direction of the rays in world coordinate
    """
    logging.info("GET RAYS MVS")
    logging.info("inputs "+str(H)+","+str(W)+","+str(intrinsic.shape)+","+str(c2w.shape) \
        +","+str(N_rays)+","+str(isRandom)+","+str(chunk)+","+str(idx))

    device = c2w.device

    # Generate pixel coordinates to construct rays to
    if variable_patches: # GRAF discriminator with scale annealing
        # Get sample indeces
        select_inds = patch_ray_sampler(patch_size, step, scale_anneal=scale_anneal)

        # Image coordinates
        coord_ws, coord_hs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))

        # Sample coordinates in patches
        select_hs = torch.nn.functional.grid_sample(coord_hs.unsqueeze(0).unsqueeze(0),
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)
        select_ws = torch.nn.functional.grid_sample(coord_ws.unsqueeze(0).unsqueeze(0),
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)

        ys, xs = select_ws.reshape(-1).int().to(device), select_hs.reshape(-1).int().to(device)

    elif N_patches:
        # When training with patches we generate <N_patches> patches
        # of patch_size*patch_size size at random positions

        # Random top left corner for patches
        xb, yb = torch.randint(0, W - patch_size, (N_patches,)), torch.randint(0, H - patch_size, (N_patches,))
        yl, xl = [], []
        for (x, y) in zip(xb, yb):
            ym, xm = torch.meshgrid(torch.linspace(y.item(), y.item() + patch_size - 1, patch_size),
                                    torch.linspace(x.item(), x.item() + patch_size - 1, patch_size))
            yl.append(ym)
            xl.append(xm)
        ys, xs = torch.cat(yl), torch.cat(xl)
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        ys, xs = ys.to(device), xs.to(device)

        if (ys < 0).any() or (ys >= H).any() or (xs < 0).any() or (xs >= W).any():
            raise ValueError("point coordinates out of bounds")

    elif isRandom:
        # When training with random ray sampling
        xs, ys = torch.randint(0,W,(N_rays,)).float().to(device), torch.randint(0,H,(N_rays,)).float().to(device)
    else:
        # When testing we generate a grid of rays
        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        if chunk>0:
            ys, xs = ys[idx*chunk:(idx+1)*chunk], xs[idx*chunk:(idx+1)*chunk]
        ys, xs = ys.to(device), xs.to(device)

    xs = xs.repeat(intrinsic.shape[0],1)
    ys = ys.repeat(intrinsic.shape[0],1)

    # Camera Coordinate homogeneous points
    dirs = torch.stack([(xs-intrinsic[:,0,2].reshape(-1,1))/intrinsic[:,0,0].reshape(-1,1),
                        (ys-intrinsic[:,1,2].reshape(-1,1))/intrinsic[:,1,1].reshape(-1,1),
                        torch.ones_like(xs)], -1)

    # World coordinate rays
    rays_d = torch.matmul(dirs, c2w[:,:3,:3].transpose(1,2))

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:,:3,-1].clone()

    # Camera pixels where rays are pointing to
    pixel_coordinates = torch.stack((ys,xs),dim=1) # row col [N, 2, N_rays]

    logging.info("outputs "+str(rays_o.shape)+","+str(rays_d.shape)+","+str(pixel_coordinates.shape))

    return rays_o, rays_d, pixel_coordinates

def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
    """
    Transform rays from world coordinate to NDC (Normalised Device Coordinates).
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    logging.info("GET NDC COORD")
    logging.info("inputs "+str(w2c_ref.shape)+","+str(intrinsic_ref.shape)+","+str(point_samples.shape) \
        +","+str(inv_scale.shape)+","+str(near)+","+str(far)+","+str(pad)+","+str(lindisp))

    N_rays, N_samples = point_samples.shape[1], point_samples.shape[2]
    point_samples = point_samples.reshape(1,-1, 3)

    # wrap to ref view
    if w2c_ref is not None:
        R = w2c_ref[:, :3, :3]  # (3, 3)
        T = w2c_ref[:, :3, 3:]  # (3, 1)
        point_samples = torch.matmul(point_samples, R.transpose(1,2)) + T.reshape(1,1,3)

    if intrinsic_ref is not None:
        # using projection
        point_samples_pixel =  point_samples @ intrinsic_ref.transpose(1,2)
        point_samples_pixel[:,:,:2] = (point_samples_pixel[:,:,:2] / point_samples_pixel[:,:,-1:] + 0.0) / inv_scale.reshape(1,1,2)  # normalize to 0~1
        if not lindisp:
            point_samples_pixel[:,:,2] = (point_samples_pixel[:,:,2] - near) / (far - near)  # normalize to 0~1
        else:
            point_samples_pixel[:,:,2] = (1.0/point_samples_pixel[:,:,2]-1.0/near)/(1.0/far - 1.0/near)
    else:
        # using bounding box
        near, far = near.view(1,1,3), far.view(1,1,3)
        point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
    del point_samples

    if pad>0:
        W_feat, H_feat = (inv_scale+1)/4.0
        point_samples_pixel[:,:,1] = point_samples_pixel[:,:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
        point_samples_pixel[:,:,0] = point_samples_pixel[:,:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)

    point_samples_pixel = point_samples_pixel.view(1, N_rays, N_samples, 3)

    logging.info("outputs "+str(point_samples_pixel.shape))
    return point_samples_pixel

def build_rays(imgs, depths, w2cs, c2ws, intrinsics, near_fars, N_samples, N_rays=1024,
               stratified=True, pad=0, chunk=-1, idx=-1, ref_idx=0, val=False,
               isRandom=True, patch_size=-1, scale_anneal=-1, step=0, variable_patches=False):
    '''

    Args:
        imgs: [N V C H W]
        depths: [N V H W]
        w2c: [N V 4 4]
        c2w: [N V 4 4]
        intrinsic: [N V 3 3]
        near_fars: [N V 2]
        N_rays: int (i.e. 1024)
        N_samples: int (i.e. 128)
        pad: int (i.e. 24)

    Returns:
        [3 N_rays N_samples]
    '''
    logging.info("BUILD RAYS")
    logging.info("inputs "+str(imgs.shape)+","+str(w2cs.shape)+","+str(c2ws.shape) \
        +","+str(intrinsics.shape)+","+str(near_fars.shape)+","+str(N_rays)+","+str(N_samples)+","+str(pad)+","+str(ref_idx))

    device = imgs.device

    N, V, C, H, W = imgs.shape
    inv_scale = torch.tensor([W-1, H-1]).to(device)

    N_patches = None
    if patch_size > 0:
        N_patches = N_rays // (patch_size * patch_size)
        assert N_rays % (patch_size * patch_size) == 0, "Batch size %d is not divisible by patch size of %d" % (N_rays, patch_size)

    # Generate ray coordinates for target camera
    c2w_tgt = c2ws[:,-1,:,:]
    intrinsic_tgt = intrinsics[:,-1,:,:]
    rays_o, rays_d, pixel_coordinates = get_rays_mvs(H, W, intrinsic_tgt, c2w_tgt,
                                                     N_rays, isRandom=isRandom,
                                                     chunk=chunk, idx=idx,
                                                     N_patches=N_patches, patch_size=patch_size,
                                                     scale_anneal=scale_anneal, step=step,
                                                     variable_patches=variable_patches)   # [N_rays 3]
    if val:
        ray_samples = H * W if chunk < 0 else pixel_coordinates.shape[-1]
    else:
        ray_samples = N_rays

    # position
    rays_o = rays_o.reshape(1, 1, 3)
    rays_o = rays_o.expand(-1, ray_samples, -1)

    pixel_coordinates_int = pixel_coordinates.long()
    color = imgs[:, -1, :, pixel_coordinates_int[0,0], pixel_coordinates_int[0,1]] # [3 N_rays] # colour of the image at the selected pixel (N_rays)
    color = color.permute(0,2,1)

    rays_depth_gt = depths[:, -1, pixel_coordinates_int[0,0], pixel_coordinates_int[0,1]]

    # travel along the rays
    near_tgt, far_tgt = near_fars[:, -1, 0], near_fars[:, -1, 1]
    t_vals = torch.linspace(0., 1., steps=N_samples).view(1,N_samples).to(device)
    depth_candidate = near_tgt * (1. - t_vals) + far_tgt * (t_vals)
    depth_candidate = depth_candidate.expand([ray_samples, N_samples])

    if stratified:
        # get intervals between samples
        mids = .5 * (depth_candidate[..., 1:] + depth_candidate[..., :-1])
        upper = torch.cat([mids, depth_candidate[..., -1:]], -1)
        lower = torch.cat([depth_candidate[..., :1], mids], -1)

        # stratified samples in those intervals
        t_rand = torch.rand(depth_candidate.shape, device=device)
        depth_candidate = lower + (upper - lower) * t_rand

    depth_candidate = depth_candidate.unsqueeze(0)
    # Samples along ray: points in space to evaluate model at
    point_samples = rays_o.unsqueeze(2) + depth_candidate.unsqueeze(-1) * rays_d.unsqueeze(2) # [ray_samples N_samples 3]

    # Transform rays to Normalised Device Coordinates
    # with respect the reference image (ref_idx=0)
    near_ref, far_ref = near_fars[:,ref_idx, 0], near_fars[:,ref_idx, 1]
    w2c_ref = w2cs[:,ref_idx,:,:]
    intrinsic_ref = intrinsics[:,ref_idx,:,:]
    points_ndc = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples,
                                    inv_scale, near=near_ref, far=far_ref, pad=pad)

    logging.info("outputs "+str(point_samples.shape)+","+str(rays_d.shape)+"," \
        +str(color.shape)+","+str(points_ndc.shape)+","+str(depth_candidate.shape)+"," \
        +str(rays_depth_gt.shape))

    return point_samples, rays_d, color, points_ndc, depth_candidate, rays_depth_gt, t_vals

def index_point_feature(volume_feature, ray_coordinate_ref):
    ''''
    Args:
        volume_color_feature: [N, G, D, h, w]
        volume_density_feature: [N C D H W]
        ray_dir_world:[3 ray_samples N_samples]
        ray_coordinate_ref:  [N N_rays N_samples 3] AKA ray_ndc?
        ray_dir_ref:  [3 N_rays]
        depth_candidates: [N_rays, N_samples]
    Returns:
        [N_rays, N_samples]
    '''
    logging.info("INDEX POINT FEATURE")
    logging.info("inputs "+str(volume_feature.shape)+","+str(ray_coordinate_ref.shape))

    device = volume_feature.device
    H, W = ray_coordinate_ref.shape[-3:-1]

    grid = ray_coordinate_ref.view(-1, 1, H,  W, 3).to(device) * 2 - 1.0  # [N, D, H, W, 3] (x,y,z) D=1

    # grid_sample: [N, C, D_in, H_in, W_in] [N, D_out, H_out, W_out, 3] -> [N, C, D_out, H_out, W_out]
    # e.g. [1, 8, 128, 176, 208] [1, 1, 1024, 128, 3] -> [1, 8, 1, 1024, 128]
    features = F.grid_sample(volume_feature, grid, align_corners=True, mode='bilinear')
    features = features[:,:,0,:,:].permute(0,2,3,1) # [N, H, W, C]

    logging.info("outputs "+str(features.shape))
    return features

def build_color_volume(point_samples, data_mvs, imgs, img_feat=None, downscale=1.0, with_mask=False):
    '''
    point_samples: [N N_ray N_sample 3]
    imgs: [N V 3 H W]
    '''
    logging.info("BUILD COLOR VOLUME")
    logging.info("inputs "+str(point_samples.shape)+","+","+str(imgs.shape) \
        +","+str(img_feat.shape if img_feat is not None else "None")+","+str(downscale)+","+str(with_mask))

    device = imgs.device
    N, V, C, H, W = imgs.shape
    inv_scale = torch.tensor([W - 1, H - 1]).to(device)

    if with_mask:
        C += 1

    if img_feat:
        C += img_feat.shape[3]

    colors = torch.empty((*point_samples.shape[:3], V*C), device=imgs.device, dtype=torch.float)

    for idx in range(V):

        w2c_ref, intrinsic_ref = data_mvs['w2cs'][:,idx,:,:], data_mvs['intrinsics'][:,idx,:,:].clone()

        point_samples_pixel = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale) # [N, N_rays, N_samples, 3]
        grid = point_samples_pixel[...,:2]*2.0-1.0 # [N, N_rays, N_samples, 2]

        # [N, 3, H, W] [N, N_rays, N_samples, 2] -> [N, 3, N_rays, N_samples] for image at view V=idx
        data = F.grid_sample(imgs[:, idx], grid, align_corners=True, mode='bilinear', padding_mode='border')

        if img_feat is not None:
            data = torch.cat((data,F.grid_sample(img_feat[:,idx], grid, align_corners=True, mode='bilinear', padding_mode='zeros')),dim=1)

        if with_mask:
            in_mask = ((grid >-1.0)*(grid < 1.0)) #in_mask[n,n_ray,n_sample,i] = is grid[n,n_ray,n_sample,i] value between -1 AND 1
            in_mask = (in_mask[...,0]*in_mask[...,1]).float() # matrix of 0 and 1
            data = torch.cat((data,in_mask.unsqueeze(1)), dim=1) # Concat along camera views

        colors[...,idx*C:idx*C+C] = data.permute(0, 2, 3, 1)

        del grid, point_samples_pixel, data

    logging.info("outputs "+str(colors.shape))
    return colors

#################################### Image tools ###############################

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
