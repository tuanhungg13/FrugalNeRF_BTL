import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import sqlite3
import sys
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

from .colmapUtils.read_write_model import *
from utils import DepthEstimator

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
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

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
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=60, random_poses=False):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        if random_poses:
            c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads * np.concatenate([2 * np.random.rand(3) - 1., [1,]]))
        else:
            c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120, N_rots=2, random_poses=False):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views, N_rots=N_rots, random_poses=random_poses)
    return np.stack(render_poses)

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    return poses, bds

def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def load_colmap_depth(datadir, factor=8, bd_factor=.75):
    data_file = datadir + '/colmap_depth.npy'
    
    images = read_images_binary(datadir + '/dense/sparse/images.bin')
    points = read_points3d_binary(datadir + '/dense/sparse/points3D.bin')
    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw = _load_data(os.path.dirname(datadir), factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)

        if len(depth_list) > 0:
            data_list.append({"name": images[id_im].name, "depth":np.array(depth_list), "coord":np.array(coord_list), "error":np.array(weight_list)})
    data_list = sorted(data_list, key=lambda x: x['name'])
    np.save(data_file, data_list)
    return data_list


class LLFFDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, frame_num=[]):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()
        self.frame_num = frame_num
        self.frame_len = len(frame_num)
        self.scene_name = datadir.split("/")[-1]
        self.dataset_name = datadir.split("/")[-2]

        self.blender2opencv = np.eye(4)
        self.read_meta()
        self.white_bg = False

        self.near_far = [0.0, 1.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def pre_calculate_nearest_pose(self, img_list):
        num_camera_pose = len(img_list)

        nearest_dist = np.full(len(self.poses), np.inf) # index; input_pose_index, output: its nearest_pose_index
        nearest_pose = np.full(len(self.poses), -1)

        dist = 0
        cur, next = -1, -1
        for i in range(num_camera_pose - 1):
            cur = img_list[i]
            for j in range(i + 1, num_camera_pose):
                next = img_list[j]
                dist = np.linalg.norm(self.poses[cur][:, 3] - self.poses[next][:, 3])
                if dist < nearest_dist[cur]:
                    nearest_dist[cur] = dist
                    nearest_pose[cur] = next
                if dist < nearest_dist[next]:
                    nearest_dist[next] = dist
                    nearest_pose[next] = cur
        return nearest_pose
    
    def get_nearest_pose(self, c2w, img_list, i):
        # calculate neighbor poses
        min_distance = -1
        for j in img_list:
            if j == i and self.split == 'train':
                continue
            distance = (torch.sum(((c2w[:3,3] - self.poses[j,:,3])**2)))**0.5
            
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                nearest_id = j
        return nearest_id

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images_4/*')))
        
        # load full resolution image then resize
        if self.split in ['train', 'test', 'novel']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        tt = self.poses[:, :3, 3]
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)
        
        if self.frame_num is not None and len(self.frame_num) > 0 and self.split != 'test':
            N_views, N_rots = 120, 1
            if self.split == 'novel':
                self.render_path = get_spiral(self.poses[self.frame_num], self.near_fars, N_views=1000, random_poses=True)
            else:
                self.render_path = get_spiral(self.poses[self.frame_num], self.near_fars, N_views=N_views, N_rots=N_rots)
            if self.split == 'novel':
                self.pose_avg = average_poses(self.poses[self.frame_num])
        else:
            N_views, N_rots = 120, 2
            if self.split == 'novel':
                self.render_path = get_spiral(self.poses, self.near_fars, N_views=1000, random_poses=True)
            else:
                self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views, N_rots=N_rots)

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        if self.frame_num is not None and len(self.frame_num) > 0:
            img_list = self.frame_num
        elif self.split == 'novel':
            if self.frame_num is not None and len(self.frame_num) > 0:
                img_list = self.frame_num
            else:
                img_list = []
        else:
            i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
            img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        # nearest_pose_ids = torch.from_numpy(self.pre_calculate_nearest_pose(img_list))
        self.all_rays = []
        self.all_rays_real = []
        self.all_rgbs = []
        self.all_ids = []
        self.all_nearest_ids = []
        self.all_depths = []
        self.all_depth_weights = []
        self.all_dense_depths = []

        if self.split == 'train':
            # Determine training images for sparse depth generation
            train_indices = img_list if self.frame_num is not None and len(self.frame_num) > 0 else img_list
            n_train = len(train_indices)
            
            # Generate sparse depth if not exists
            depth_dir = self.root_dir + "/" + str(n_train) + "_views"
            if not os.path.exists(os.path.join(depth_dir, 'colmap_depth.npy')):
                print(f"Generating sparse depth for {n_train} training views...")
                work_dir = generate_sparse_depth(self.root_dir, train_indices, self.downsample)
                if work_dir is None:
                    print("Failed to generate sparse depth. Using empty depth data.")
                    self.depth_gts = []
                else:
                    self.depth_gts = load_colmap_depth(depth_dir, factor=self.downsample)
            else:
                print(f"Loading existing sparse depth from {depth_dir}")
                self.depth_gts = load_colmap_depth(depth_dir, factor=self.downsample)
        
        if self.split != 'novel':
            self.frameid2_startpoints_in_allray = [-10] * self.poses.shape[0] # -10 represent
            cnt = 0
            depth_estimator = DepthEstimator()
            for index, i in enumerate(img_list):
                image_path = self.image_paths[i]
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                if self.downsample != 1.0:
                    img = img.resize(tuple(self.img_wh), Image.LANCZOS)
                
                img = self.transform(img)  # (3, h, w)
                
                depth = -torch.ones(H, W)
                dense_depth = depth_estimator.estimate_depth(img.cuda()).cpu()
                weight = -torch.ones(H, W)
                if self.split == 'train':
                    for j in range(len(self.depth_gts[index]['coord'])):
                        # avoid out of bound
                        x = round(self.depth_gts[index]['coord'][j,1]) 
                        x = x if x < H else H-1
                        y = round(self.depth_gts[index]['coord'][j,0])
                        y = y if y < W else W-1
                        depth[x, y] = self.depth_gts[index]['depth'][j]
                        weight[x, y] = self.depth_gts[index]['error'][j]

                depth = depth.view(-1)
                weight = weight.view(-1)
                dense_depth = dense_depth.view(-1)

                nearest_id = self.get_nearest_pose(c2w, img_list, i)
                                
                        

                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]
                # self.all_view_ids += [id]
                self.all_depths += [depth]
                self.all_depth_weights += [weight]
                self.all_dense_depths += [dense_depth]
                ray_real_o, rays_real_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, ray_real_o, rays_real_d)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                self.all_rays_real += [torch.cat([ray_real_o, rays_real_d], 1)]
                cur_ids = torch.full([rays_o.shape[0]], i)
                self.all_ids += [cur_ids]
                self.all_nearest_ids += [torch.ones_like(cur_ids).int() * nearest_id]
                self.frameid2_startpoints_in_allray[i] = cnt * cur_ids.shape[0] - 1
                cnt += 1 
        
        if self.split == 'novel':
            cnt = 0
            self.frameid2_startpoints_in_allray = [-10] * self.render_path.shape[0]
            for i, c2w in enumerate(self.render_path):
                c2w = torch.FloatTensor(c2w)
                ray_real_o, rays_real_d = get_rays(self.directions, c2w)
                rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, ray_real_o, rays_real_d)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                self.all_rays_real += [torch.cat([ray_real_o, rays_real_d], 1)]
                cur_ids = torch.full([rays_o.shape[0]], i)
                self.all_ids += [cur_ids]
                nearest_id = self.get_nearest_pose(c2w, img_list, i)
                self.all_nearest_ids += [torch.ones_like(cur_ids).int() * nearest_id]
                self.frameid2_startpoints_in_allray[i] = cnt * cur_ids.shape[0] - 1
                cnt += 1
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rays_real = torch.cat(self.all_rays_real, 0)
            self.all_ids = torch.cat(self.all_ids, 0).to(torch.int)
            self.all_nearest_ids = torch.cat(self.all_nearest_ids, 0).to(torch.int)
        else:
            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
                self.all_rays_real = torch.cat(self.all_rays_real, 0)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
                self.all_depths = torch.cat(self.all_depths, 0)
                self.all_depth_weights = torch.cat(self.all_depth_weights, 0)
                self.all_dense_depths = torch.cat(self.all_dense_depths, 0)
                self.all_ids = torch.cat(self.all_ids, 0).to(torch.int)
                self.all_nearest_ids = torch.cat(self.all_nearest_ids, 0).to(torch.int)
            else:
                self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames']),h,w, 3)
                self.all_rays_real = torch.stack(self.all_rays_real, 0)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames']),h,w,3)
                self.all_depths = torch.stack(self.all_depths, 0).reshape(-1,*self.img_wh[::-1], 1)
                self.all_depth_weights = torch.stack(self.all_depth_weights, 0).reshape(-1,*self.img_wh[::-1], 1)
                self.all_dense_depths = torch.stack(self.all_dense_depths, 0).reshape(-1,*self.img_wh[::-1], 1)
                self.all_ids = torch.stack(self.all_ids, 0).to(torch.int)
                self.all_nearest_ids = torch.stack(self.all_nearest_ids, 0).to(torch.int)
                masks_np = np.ones((len(self.frame_num), self.img_wh[1], self.img_wh[0], 1), dtype=bool)
                self.all_masks = torch.from_numpy(masks_np).to(torch.bool)
                self.all_masks = self.all_masks.reshape(-1, self.img_wh[1], self.img_wh[0], 1)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def array_to_blob(array):
    IS_PYTHON3 = sys.version_info[0] >= 3
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    IS_PYTHON3 = sys.version_info[0] >= 3
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        
        CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL)"""
        
        CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
        """.format(2**31 - 1)
        
        CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
        """
        
        CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
        
        CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB)"""
        
        CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB)
        """
        
        CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
        
        CREATE_ALL = "; ".join([
            CREATE_CAMERAS_TABLE,
            CREATE_IMAGES_TABLE,
            CREATE_KEYPOINTS_TABLE,
            CREATE_DESCRIPTORS_TABLE,
            CREATE_MATCHES_TABLE,
            CREATE_TWO_VIEW_GEOMETRIES_TABLE,
            CREATE_NAME_INDEX
        ])
        
        self.create_tables = lambda: self.executescript(CREATE_ALL)

def generate_sparse_depth(datadir, frame_indices, downsample=4):
    """
    Generate sparse depth using COLMAP pipeline for selected frame indices
    Args:
        datadir: root directory containing images and poses_bounds.npy
        frame_indices: list of frame indices to use for sparse reconstruction
        downsample: downsample factor for images
    """
    n_views = len(frame_indices)
    work_dir = os.path.join(datadir, f"{n_views}_views")
    
    # Create working directory structure
    if os.path.exists(work_dir):
        os.system(f'rm -rf {work_dir}')
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'created'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'triangulated'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'images'), exist_ok=True)
    
    # Check if sparse reconstruction exists
    sparse_dir = os.path.join(datadir, 'sparse', '0')
    if not os.path.exists(sparse_dir):
        # Try alternative sparse directory locations
        alt_sparse_dirs = [
            os.path.join(datadir, 'sparse'),
            os.path.join(datadir, 'colmap_sparse', '0'),
            os.path.join(datadir, 'colmap_sparse')
        ]
        found_sparse = False
        for alt_dir in alt_sparse_dirs:
            if os.path.exists(alt_dir):
                sparse_dir = alt_dir
                found_sparse = True
                print(f"Found sparse reconstruction at {sparse_dir}")
                break
        
        if not found_sparse:
            print(f"Warning: No sparse reconstruction found at {sparse_dir}")
            print("Checked alternative locations:", alt_sparse_dirs)
            print("Please run COLMAP first to generate initial sparse reconstruction")
            return None
    
    # Convert sparse model to TXT format if needed
    if not os.path.exists(os.path.join(sparse_dir, 'images.txt')):
        os.system(f'colmap model_converter --input_path {sparse_dir} --output_path {sparse_dir} --output_type TXT')
    
    # Read original poses and images
    poses_bounds = np.load(os.path.join(datadir, 'poses_bounds.npy'))
    all_image_paths = sorted(glob.glob(os.path.join(datadir, f'images_{downsample}/*')))
    if len(all_image_paths) == 0:
        # Try alternative image directory naming
        all_image_paths = sorted(glob.glob(os.path.join(datadir, 'images/*')))
    
    # Read COLMAP images
    images = {}
    images_txt_path = os.path.join(sparse_dir, 'images.txt')
    if os.path.exists(images_txt_path):
        with open(images_txt_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    image_id = int(elems[0])
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    camera_id = int(elems[8])
                    image_name = elems[9]
                    fid.readline()  # Skip next line
                    images[image_name] = elems[1:]
    
    # Select and copy training images based on frame_indices
    selected_images = []
    print(f"Found {len(all_image_paths)} total images")
    print(f"Selected frame indices: {frame_indices}")
    
    for idx in frame_indices:
        if idx < len(all_image_paths):
            img_path = all_image_paths[idx]
            img_name = os.path.basename(img_path)
            selected_images.append(img_name)
            # Copy image to working directory
            dst_path = os.path.join(work_dir, 'images', img_name)
            os.system(f'cp "{img_path}" "{dst_path}"')
            print(f"Copied image {idx}: {img_name}")
        else:
            print(f"Warning: Frame index {idx} is out of range (max: {len(all_image_paths)-1})")
    
    print(f"Selected {len(selected_images)} images for sparse reconstruction")
    
    # Copy camera parameters
    cameras_txt = os.path.join(sparse_dir, 'cameras.txt')
    if os.path.exists(cameras_txt):
        os.system(f'cp "{cameras_txt}" "{os.path.join(work_dir, "created", "cameras.txt")}"')
    
    # Create empty points3D.txt
    with open(os.path.join(work_dir, 'created', 'points3D.txt'), "w") as fid:
        pass
    
    # Change to working directory for COLMAP operations
    original_dir = os.getcwd()
    os.chdir(work_dir)
    
    try:
        # Feature extraction
        print("Running COLMAP feature extraction...")
        os.system('colmap feature_extractor --database_path database.db --image_path images '
                 '--SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 '
                 '--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --ImageReader.single_camera 1')
        
        # Feature matching
        print("Running COLMAP feature matching...")
        os.system('colmap exhaustive_matcher --database_path database.db '
                 '--SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
        
        # Get image order from database
        try:
            db = COLMAPDatabase.connect('database.db')
            db_images = db.execute("SELECT * FROM images")
            img_rank = [db_image[1] for db_image in db_images]
            db.close()
            print(f"Found {len(img_rank)} images in database")
        except Exception as e:
            print(f"Error reading database: {e}")
            img_rank = selected_images
        
        # Create images.txt for triangulation
        with open('created/images.txt', "w") as fid:
            for idx, img_name in enumerate(img_rank):
                if os.path.basename(img_name) in images:
                    data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
                    fid.writelines(data)
        
        # Point triangulation
        print("Running COLMAP point triangulation...")
        os.system('colmap point_triangulator --database_path database.db --image_path images '
                 '--input_path created --output_path triangulated '
                 '--Mapper.tri_ignore_two_view_tracks 0 --Mapper.num_threads 16 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0')
        
        # Convert to TXT format
        os.system('colmap model_converter --input_path triangulated --output_path triangulated --output_type TXT')
        
        # Image undistortion
        print("Running COLMAP image undistortion...")
        os.system('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')
        
        print(f"Sparse depth generation completed for {n_views} views")
        
        # Verify that dense directory was created successfully
        dense_dir = os.path.join(work_dir, 'dense')
        if os.path.exists(dense_dir) and os.path.exists(os.path.join(dense_dir, 'sparse')):
            print(f"Dense reconstruction created successfully at {dense_dir}")
            return work_dir
        else:
            print(f"Warning: Dense reconstruction may not have been created properly")
            return work_dir
            
    except Exception as e:
        print(f"Error during COLMAP processing: {e}")
        return None
    finally:
        # Return to original directory
        os.chdir(original_dir)