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

import logging

import torch
import torch.nn.functional as F

from utils import index_point_feature, build_color_volume

def gen_dir_feature(w2c_ref, rays_dir):
    """
    Inputs:
        w2c_ref: [N,4,4]
        rays_dir: [N, N_rays, 3]

    Returns:
        dirs: [N, N_rays, 3]
    """
    logging.info("GEN DIR FEATURE")
    logging.info("inputs "+str(w2c_ref.shape)+","+str(rays_dir.shape))

    dirs = rays_dir @ w2c_ref[:,:3,:3].transpose(1,2) # [N, N_rays, 3]

    logging.info("outputs "+str(dirs.shape))
    return dirs

def gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, feat_dim, img_feat=None, img_downscale=1.0, use_color_volume=False, net_type='v0'):
    logging.info("GEN PTS FEATS")
    logging.info("inputs "+str(imgs.shape)+","+str(volume_feature.shape)+"," \
        +str(rays_pts.shape) +","+str(rays_ndc.shape)+","+str(feat_dim)+"," \
        +str(img_feat.shape if img_feat is not None else "None")+"," \
        +str(img_downscale)+","+str(use_color_volume)+","+str(net_type))

    N, N_rays, N_samples = rays_pts.shape[:3]

    if img_feat is not None:
        feat_dim += img_feat.shape[2]*img_feat.shape[3]

    if not use_color_volume:
        input_feat = torch.empty((N, N_rays, N_samples, feat_dim), device=imgs.device, dtype=torch.float)
        ray_feats = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
        input_feat[..., :8] = ray_feats
        input_feat[..., 8:] = build_color_volume(rays_pts, pose_ref, imgs, img_feat, with_mask=True, downscale=img_downscale)
    else:
        input_feat = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)

    logging.info("outputs "+str(input_feat.shape))
    return input_feat

def depth2dist(z_vals, cos_angle):
    """
    z_vals: [N_ray N_sample]
    """
    device = z_vals.device

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * cos_angle.unsqueeze(-1)
    return dists

def raw2alpha(sigma):
    """
    Function for computing density from model prediction.
    This value is strictly between [0, 1].
    """
    logging.info("RAW2ALPHA")
    logging.info("inputs "+str(sigma.shape))

    alpha = 1. - torch.exp(-sigma)

    # Transmission
    # Compute weight for RGB of each sample along each ray. 
    # A cumprod() is used to express the idea of the ray
    # not having reflected up to this point yet.
    # [N_rays, N_samples]
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], alpha.shape[1], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:,:, :-1]
    # Alpha composite weights
    weights = alpha * T  # [N, N_rays, N_samples]

    logging.info("outputs "+str(alpha.shape)+","+str(weights.shape))
    return alpha, weights

def raw2outputs(raw, z_vals, dists, white_bkgd=False, net_type='v2'):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [N, N_rays, N_samples, 4]. Prediction from model.
        z_vals: [N, N_rays, N_samples]. Integration time. (depth_candidates)
        dists:  [N, N_rays, N_samples]. Distances. (depth2dist) (t_{i+1}-t_i)
    Returns:
        rgb_map:   [N, N_rays, 3]. Estimated RGB color of a ray.
        disp_map:  [N, N_rays]. Disparity map. Inverse of depth map.
        acc_map:   [N, N_rays]. Sum of weights along each ray.
        weights:   [N, N_rays, N_samples]. Weights assigned to each sampled color.
        depth_map: [N, N_rays]. Estimated distance to object.
    """
    logging.info("RAW2OUTPUTS")
    logging.info("inputs "+str(raw.shape)+","+str(z_vals.shape)+"," \
        +str(dists.shape)+","+str(white_bkgd)+","+str(net_type))

    device = z_vals.device

    rgb = raw[..., :3] # [N, N_rays, N_samples, 3] rgb for each sample point along the ray

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha, weights = raw2alpha(raw[..., 3])  # [N, N_rays, N_samples]

    # Computed weighted color of each sample along each ray.
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N, N_rays, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, -1) # [N, N_rays]

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, -1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    logging.info("outputs "+str(rgb_map.shape)+","+str(disp_map.shape)+"," \
        +str(acc_map.shape)+","+str(weights.shape)+","+str(depth_map.shape)+","+str(alpha.shape))
    return rgb_map, disp_map, acc_map, weights, depth_map, alpha

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    logging.info("BATCHIFY")
    logging.info("inputs "+str(chunk))

    if chunk is None:
        return fn

    def ret(inputs, alpha_only):
        if alpha_only:
            return torch.cat([fn.forward_alpha(inputs[:,i:i + chunk]) for i in range(0, inputs.shape[1], chunk)], 1)
        else:
            return torch.cat([fn(inputs[:,i:i + chunk]) for i in range(0, inputs.shape[1], chunk)], 1)

    return ret

def rendering(args, data_mvs, rays_pts, rays_ndc, depth_candidates, rays_dir,
              volume_feature=None, imgs=None, img_feat=None, network_fn=None,
              embedding_pts=None, embedding_dir=None, white_bkgd=False):
    """
    rays_dir: [N, N_rays, 3] (e.g. [N,1024,3])
    """
    logging.info("RENDERING")
    logging.info("inputs "+str(rays_pts.shape)+","+str(rays_ndc.shape) \
        +","+str(depth_candidates.shape)+","+str(rays_dir.shape) \
        +","+str(volume_feature.shape if volume_feature is not None else "None") \
        +","+str(imgs.shape if imgs is not None else "None") \
        +","+str(img_feat.shape if img_feat is not None else "None") \
        +","+str("network_fn" if network_fn is not None else "None") \
        +","+str("embedding_pts" if embedding_pts is not None else "None") \
        +","+str("embedding_dir" if embedding_dir is not None else "None") \
        +","+str(white_bkgd))

    # rays angle
    cos_angle = torch.norm(rays_dir, dim=-1) # [N, N_rays]

    # using direction
    if data_mvs is not None:
        w2ref = data_mvs['w2cs'][:,0,:,:]
        angle = gen_dir_feature(w2ref, rays_dir/cos_angle.unsqueeze(-1))  # view dir feature
    else:
        angle = rays_dir/cos_angle.unsqueeze(-1) # [N, N_rays, 1]

    # rays_pts
    input_feat = gen_pts_feats(imgs, volume_feature, rays_pts, data_mvs,
                               rays_ndc, args.feat_dim, img_feat,
                               args.img_downscale, args.use_color_volume, args.net_type)

    pts = rays_ndc
    if embedding_pts:
        pts = embedding_pts(rays_ndc)

    if input_feat is not None:
        pts = torch.cat((pts, input_feat), dim=-1)

    if angle is not None:
        if angle.dim()!=4:
            # Expand angle (view direction) for each sample
            angle = angle.unsqueeze(dim=2).expand(-1,-1,pts.shape[2],-1)

        if embedding_dir is not None:
            angle = embedding_dir(angle)

        pts = torch.cat([pts, angle], -1)

    alpha_only = angle is None
    outputs_flat = batchify(network_fn, args.netchunk)(pts, alpha_only) # Apply nerf to points by chunks

    raw = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    # raw = all the colours and densities of every sample point at every ray

    if raw.shape[-1]>4:
        input_feat = torch.cat((input_feat[...,:8],raw[...,4:]), dim=-1)

    dists = depth2dist(depth_candidates, cos_angle)

    rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw,
                                                                        depth_candidates,
                                                                        dists,
                                                                        white_bkgd,
                                                                        args.net_type)

    logging.info("outputs "+str(rgb_map.shape)+","+str(input_feat.shape)+"," \
        +str(weights.shape)+","+str(depth_map.shape)+","+str(alpha.shape))
    return rgb_map, input_feat, weights, depth_map, alpha