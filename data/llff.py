from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
from PIL import Image
from torchvision import transforms as T

from .data_utils import get_nearest_pose_ids
from utils import read_pfm


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg



def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)z


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, root_dir, config_dir, split='train', spheric_poses=True,
                 load_ref=False, downSample=1.0, pair_idx=None, max_len=-1,
                 scene=None, depth_path=None, closest_views=False):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = Path(root_dir)
        self.config_dir = Path(config_dir)
        self.split = split
        self.downSample = downSample
        self.img_wh = (int(960*downSample),int(640*downSample))
        assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
            'image width must be divisible by 32, you may need to modify the imgScale'
        self.spheric_poses = spheric_poses
        self.max_len = max_len
        self.closest_views = closest_views
        self.define_transforms()
        self.blender2opencv = np.array([[1, 0, 0, 0],
                                        [0,-1, 0, 0],
                                        [0, 0,-1, 0],
                                        [0, 0, 0, 1]])

        self.build_metas(scene)
        self.build_proj_mats()
        self.white_back = False

        self.scale_factor = 1.0 / 200 # scale factor for DTU depth maps
        depth_path = Path(depth_path) if depth_path else None
        self.depth_files = sorted(depth_path.glob('**/*.pfm')) if depth_path else []


    def build_metas(self, scene):
        if scene == None:
            scene_list = self.config_dir / f'lists/llff_{self.split}_all.txt'

            with open(scene_list) as f:
                self.scenes = [line.rstrip() for line in f.readlines()]
        else:
            self.scenes = [scene]

        self.image_paths = {}
        self.metas = []
        for scene in self.scenes:
            scene_path = self.root_dir / scene
            self.image_paths[scene] = sorted(scene_path.glob('**/images_4/*'))
            view_ids = list(range(len(self.image_paths[scene])))
            for id in view_ids:
                self.metas += [(scene, id, view_ids.copy())]

    def build_proj_mats(self):
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = {}, {}, {}, {}
        self.poses, self.bounds = {}, {}
        self.N_images = 0
        for scene in self.scenes:

            scene_path = self.root_dir / scene
            poses_bounds = np.load(scene_path / 'poses_bounds.npy') # For each camera of scene

            if self.split in ['train', 'val']:
                assert len(poses_bounds) == len(self.image_paths[scene]), \
                    f'Mismatch between number of images {len(self.image_paths[scene])} and ' \
                        f'number of poses {len(poses_bounds)} in {scene}! Please rerun COLMAP!'

            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
            bounds = poses_bounds[:, -2:]  # (N_images, 2)

            # Step 1: rescale focal length according to training resolution
            H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

            focal = [focal* self.img_wh[0] / W, focal* self.img_wh[1] / H]

            # Step 2: correct poses
            poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            poses, _ = center_poses(poses, self.blender2opencv)

            # Step 3: correct scale so that the nearest depth is at a little more than 1.0
            near_original = bounds.min()
            scale_factor = near_original * 0.75  # 0.75 is the default parameter
            bounds /= scale_factor
            poses[..., 3] /= scale_factor
            self.bounds[scene] = bounds
            self.poses[scene] = poses

            proj_mats_scene = []
            intrinsics_scene = []
            world2cams_scene = []
            cam2worlds_scene = []

            w, h = self.img_wh
            for idx in range(len(poses)):
                # camera-to-world, world-to-camera
                c2w = torch.eye(4).float()
                c2w[:3] = torch.FloatTensor(poses[idx])
                w2c = torch.inverse(c2w)
                cam2worlds_scene.append(c2w)
                world2cams_scene.append(w2c)

                # Intrisics are the same for all views
                intrinsic = torch.tensor([[focal[0], 0, w / 2], [0, focal[1], h / 2], [0, 0, 1]]).float()
                intrinsics_scene.append(intrinsic.clone())
                intrinsic[:2] = intrinsic[:2] / 4   # 4 times downscale in the feature space

                # Projection matrices
                proj_mat_l = torch.eye(4)
                proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
                proj_mats_scene.append(proj_mat_l)
            self.proj_mats[scene] = torch.stack(proj_mats_scene).float()
            self.intrinsics[scene] = torch.stack(intrinsics_scene).float()
            self.world2cams[scene] = torch.stack(world2cams_scene).float()
            self.cam2worlds[scene] = torch.stack(cam2worlds_scene).float()

    def define_transforms(self):
        self.transform = T.ToTensor()
        self.src_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                        ])

    def read_depth(self, filename, img_wh):
        print(filename, img_wh)
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        print("depth_h shape", depth_h.shape)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        print("depth_h shape", depth_h.shape)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        print("depth_h shape", depth_h.shape)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample,
                             interpolation=cv2.INTER_NEAREST)
        print("depth_h shape", depth_h.shape)
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)
        mask = depth > 0

        depth_h = cv2.resize(depth_h, img_wh)
        print("depth_h shape", depth_h.shape)

        return depth, mask, depth_h

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        scene, target_view, src_views = self.metas[idx]

        # Returns a list of all camera poses ordered from nearest to farthest
        nearest_pose_ids = get_nearest_pose_ids(self.cam2worlds[scene][target_view],
                                                self.cam2worlds[scene],
                                                len(self.cam2worlds[scene]),
                                                tar_id=target_view,
                                                angular_dist_method='dist')

        if self.closest_views:
            # Get nearest views to the target image
            nearest_pose_ids = nearest_pose_ids[:5]
        else:
            # Get far views with re. target image
            nearest_pose_ids = nearest_pose_ids[-10:]

        # Select views
        if self.split=='train':
            ids = torch.randperm(5)[:3]
            view_ids = [nearest_pose_ids[i] for i in ids] + [target_view]
        else:
            view_ids = [nearest_pose_ids[i] for i in range(3)] + [target_view]
        print(f"Selecting cam views {view_ids}")

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        depths_h = [] # Real depth maps from unrelated dataset
        near_fars = []
        near_far_source = torch.Tensor([self.bounds[scene][view_ids].min()*0.8, self.bounds[scene][view_ids].max()*1.2])

        for i, vid in enumerate(view_ids):
            intrinsics.append(self.intrinsics[scene][vid])
            w2cs.append(self.world2cams[scene][vid])
            c2ws.append(self.cam2worlds[scene][vid])
            near_fars.append(near_far_source)

            proj_mat_ls = self.proj_mats[scene][vid]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

            # Get image
            img = Image.open(self.image_paths[scene][vid]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            imgs.append(self.src_transform(img))

            # Get depth map
            if len(self.depth_files) > 0:
                depth_filename = random.choice(self.depth_files)
                depth, mask, depth_h = self.read_depth(depth_filename, self.img_wh)
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((self.img_wh[1], self.img_wh[0])))

        sample = {}
        sample['images'] = torch.stack(imgs).float()  # (V, 3, H, W)
        sample['depths_h'] = torch.from_numpy(np.stack(depths_h)).float() # (V, H, W)
        sample['w2cs'] = torch.stack(w2cs).float()  # (V, 4, 4)
        sample['c2ws'] = torch.stack(c2ws).float()  # (V, 4, 4)
        sample['near_fars'] = torch.stack(near_fars).float()
        sample['proj_mats'] = torch.from_numpy(np.stack(proj_mats)[:,:3]).float()
        sample['intrinsics'] = torch.stack(intrinsics).float()  # (V, 3, 3)

        return sample