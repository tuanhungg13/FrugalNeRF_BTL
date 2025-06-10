import torch
import cv2 as cv
import numpy as np
import os
import sqlite3
import sys
from glob import glob
from .ray_utils import *
from torch.utils.data import Dataset
from .colmapUtils.read_write_model import *
from utils import DepthEstimator

import pandas

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

def generate_sparse_depth(datadir, frame_indices, downsample=1):
    """
    Generate sparse depth using COLMAP pipeline for selected frame indices
    Args:
        datadir: root directory containing images
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
        print(f"Warning: No sparse reconstruction found at {sparse_dir}")
        print("DTU dataset doesn't require prior sparse reconstruction - will create from scratch")
        # For DTU, we can proceed without existing sparse reconstruction
        images = {}
    else:
        # Convert sparse model to TXT format if needed
        if not os.path.exists(os.path.join(sparse_dir, 'images.txt')):
            os.system(f'colmap model_converter --input_path {sparse_dir} --output_path {sparse_dir} --output_type TXT')
        
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
    
    # Read DTU images
    all_image_paths = sorted(glob(os.path.join(datadir, 'image/*.png')))
    if len(all_image_paths) == 0:
        all_image_paths = sorted(glob(os.path.join(datadir, 'images/*.png')))
    
    print(f"Found {len(all_image_paths)} total images")
    print(f"Selected frame indices: {frame_indices}")
    
    # Select and copy training images based on frame_indices
    for idx in frame_indices:
        if idx < len(all_image_paths):
            img_path = all_image_paths[idx]
            img_name = os.path.basename(img_path)
            # Copy image to working directory
            dst_path = os.path.join(work_dir, 'images', img_name)
            os.system(f'cp "{img_path}" "{dst_path}"')
    
    # Copy camera parameters if they exist
    cameras_txt = os.path.join(sparse_dir, 'cameras.txt')
    if os.path.exists(cameras_txt):
        os.system(f'cp "{cameras_txt}" "{os.path.join(work_dir, "created", "cameras.txt")}"')
    else:
        # Create a basic camera file for DTU
        with open(os.path.join(work_dir, 'created', 'cameras.txt'), "w") as fid:
            fid.write("# Camera list with one line of data per camera:\n")
            fid.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            fid.write("1 PINHOLE 400 300 200 200 200 150\n")
    
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
        
                # Create images.txt for triangulation - simplified approach for DTU
        with open('created/images.txt', "w") as fid:
            # Use the actual copied images instead of relying on database order
            copied_images = sorted(os.listdir('images'))
            for idx, img_name in enumerate(copied_images):
                # Create a simple pose entry for each image
                fid.write(f"{idx + 1} 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 {img_name}\n\n")
            print(f"Created images.txt with {len(copied_images)} images for DTU reconstruction")
        
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

def load_sparse_depth(dataset_name, scene_name, frame_len, frame_num, resolution_suffix):
    """_summary_

    Args:
        scene_name (str): name of the scene ex. horns
        frame_num (int): frame id
        resolution_suffix (int): down sample num

    Returns:
        _type_: _description_
    """
    if dataset_name == "nerf_llff_data":
        depth_path = f'../../../estimated_depth_more_features/{dataset_name}/DE0{frame_len}/{scene_name}/estimated_depths_down{resolution_suffix}/{frame_num:04}.csv'
        depth_data = pandas.read_csv(depth_path)
        return depth_data  
    elif dataset_name == "dtu":
        scene_name = int(scene_name[4:])
        depth_path = f'/project/linjohnss/estimated_depths/DTU/DE0{frame_len}/{scene_name:05}/estimated_depths/{frame_num:04}.csv'
        depth_data = pandas.read_csv(depth_path)
        return depth_data  

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def generate_spiral_path_dtu(poses, n_frames=120, n_rots=4, zrate=.5, perc=60, random_poses=False):
    """Calculates a forward facing spiral path for rendering for DTU."""

    # Get radii for spiral path using 60th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), perc, 0) * 0.5
    radii = np.concatenate([radii, [1.]]) 

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        if random_poses:
            t *= np.concatenate([2 * np.random.rand(3) - 1., [1,]])
        position = cam2world @ t
        render_poses.append(viewmatrix(z_axis, up, position, True))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses

def rescale_poses(poses):
    """Rescales camera poses according to maximum x/y/z value."""
    s = np.max(np.abs(poses[:, :3, -1]))
    out = np.copy(poses)
    out[:, :3, -1] /= s
    return out, s

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
    return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)

def recenter_poses(poses):
    """Recenter poses around the origin."""
    
    cam2world = poses_avg(poses)
    poses = np.linalg.inv(pad_poses(cam2world)) @ poses
    return poses

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
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) #* sc
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


class DTUDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1, is_stack=False, frame_num=None, hold_every=8):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.split = split
        self.root_dir = datadir
        self.is_stack = is_stack
        self.downsample = downsample
        self.white_bg = True
        self.camera_dict = np.load(os.path.join(self.root_dir, 'cameras.npz'))

        self.img_wh = (int(400 / downsample), int(300 / downsample))
        self.hold_every = hold_every
        
        self.near_far = np.array([0.5, 3.5])
        self.frame_num = frame_num
        self.scene_name = datadir.split("/")[-1]
        self.dataset_name = "dtu"
        self.read_meta()
        self.get_bbox()

    def get_bbox(self):
        object_bbox_min = np.array([-0.8, -0.8, -0.8, 0.8])
        object_bbox_max = np.array([ 0.8,  0.8,  0.8, 0.8])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.root_dir, 'cameras.npz'))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.scene_bbox = torch.from_numpy(np.stack((object_bbox_min[:3, 0],object_bbox_max[:3, 0]))).float()

    def gen_rays_at(self, intrinsic, c2w, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        W,H = self.img_wh
        tx = torch.linspace(0, W - 1, W // l)+0.5
        ty = torch.linspace(0, H - 1, H // l)+0.5
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        intrinsic_inv = torch.inverse(intrinsic)
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(c2w[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = c2w[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1).reshape(-1,3), rays_v.transpose(0, 1).reshape(-1,3)

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
                dist = np.linalg.norm(self.poses[cur][:3, 3] - self.poses[next][:3, 3])
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
            if j == i:
                continue
            
            distance = (torch.sum(((c2w[:3,3] - self.poses[j,:3,3])**2)))**0.5
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                nearest_id = j
        return nearest_id

    def read_meta(self):
        
        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        images_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in images_lis]) / 255.0

        rgbs = torch.from_numpy(images_np.astype(np.float32)[...,[2,1,0]])  # [n_images, H, W, 3]
        self.img_wh = [rgbs.shape[2],rgbs.shape[1]]
        W,H = self.img_wh

        # world_mat is a projection matrix from world to image
        n_images = len(images_lis) 
        world_mats_np = [self.camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.scale_mats_np = [self.camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        
        # load intrinsics & poses from all imgs
        self.intrinsics, self.poses = [],[]
        for img_idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, world_mats_np)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, c2w = load_K_Rt_from_P(None, P)
            c2w = torch.from_numpy(c2w).float()
            intrinsic = torch.from_numpy(intrinsic).float()
            intrinsic[:2] /= self.downsample
            
            self.poses.append(c2w)
            self.intrinsics.append(intrinsic)
        self.intrinsics, self.poses = np.stack(self.intrinsics), np.stack(self.poses)
        self.poses, scale_factor = rescale_poses(self.poses)

        # load img list 
        if self.frame_num is not None and len(self.frame_num) > 0:
            img_list = self.frame_num
            self.frame_len = len(img_list)
        else:
            # For DTU, if no specific frames are specified, use all frames
            img_list = list(range(len(images_lis)))
            self.frame_len = len(img_list)
        
        # build rendering path
        N_views = 120
        #### 2 view spiral path smapling
        if self.frame_num is not None and len(self.frame_num) > 0 and self.split != 'test':
            if self.split == 'novel':
                self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :][self.frame_num], n_frames=1000, random_poses=True)
            else:
                self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :][self.frame_num], n_frames=N_views, n_rots=2, zrate=0.05)
        else:
            if self.split == 'novel':
                self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :], n_frames=1000, random_poses=True)
            else:
                self.render_path = generate_spiral_path_dtu(self.poses[:, :3, :], n_frames=N_views, n_rots=1, zrate=0)
            
        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.poses = torch.from_numpy(self.poses).float()
        self.all_rays = []
        self.all_rgbs = []
        self.all_ids = []
        self.all_nearest_ids = []
        self.all_depths = []
        self.all_depth_weights = []
        self.all_dense_depths = []

        if self.split == 'train':
            # Determine training images for sparse depth generation
            train_indices = self.frame_num if self.frame_num is not None and len(self.frame_num) > 0 else img_list
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
            self.frameid2_startpoints_in_allray = [-10] * self.poses.shape[0] # -10 represent None
            cnt = 0
            depth_estimator = DepthEstimator()
            for index, i in enumerate(img_list):
                c2w = self.poses[i]
                intrinsic = self.intrinsics[i]

                rays_o, rays_d = self.gen_rays_at(intrinsic,c2w)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                self.all_rgbs += [rgbs[i].reshape(-1, 3)]
                # convert rgbs[i] to tenor and reshape to (3, H, W)
                img = torch.from_numpy(images_np[i].astype(np.float32)).permute(2, 0, 1)

                # get sparse depth from csv (generated by ViP-NeRF saprse depth genrator)
                depth = -torch.ones(H, W)
                dense_depth = depth_estimator.estimate_depth(img.cuda()).cpu()
                if self.split == "train":
                    SD = load_sparse_depth(self.dataset_name, self.scene_name, self.frame_len, i, int(self.downsample))
                    for j in range(len(SD)):
                        depth[round(SD.y[j]), round(SD.x[j])] = SD.depth[j] / scale_factor
                depth = depth.view(-1)

                dense_depth = dense_depth.view(-1)

                self.all_depths += [depth]
                self.all_dense_depths += [dense_depth]
                # get nearest frame of current frame
                nearest_id = self.get_nearest_pose(c2w, img_list, i)
                cur_ids = torch.full([rays_o.shape[0]], i)
                self.all_ids += [cur_ids]
                self.all_nearest_ids += [torch.ones_like(cur_ids).int() * nearest_id]
                self.frameid2_startpoints_in_allray[i] = cnt * cur_ids.shape[0] - 1
                cnt += 1 

        # change poses shape from (n, 4, 4) to (n, 3, 4)
        self.poses = self.poses[:, :3, :]
        
        if self.split == 'novel':
            cnt = 0
            self.frameid2_startpoints_in_allray = [-10] * self.render_path.shape[0]
            for i,  c2w in enumerate(self.render_path):
                c2w = torch.FloatTensor(c2w)
                nearest_id = self.get_nearest_pose(c2w, img_list, i)
                intrinsic = self.intrinsics[nearest_id] # use nearest camera's intrinsic
                # get rays
                rays_o, rays_d = self.gen_rays_at(intrinsic,c2w)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                # get nearest training frame of current frame
                cur_ids = torch.full([rays_o.shape[0]], i)
                self.all_ids += [cur_ids]
                self.all_nearest_ids += [torch.ones_like(cur_ids).int() * nearest_id]
                self.frameid2_startpoints_in_allray[i] = cnt * cur_ids.shape[0] - 1
                cnt += 1
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rays_real = self.all_rays
            self.all_ids = torch.cat(self.all_ids, 0).to(torch.int)
            self.all_nearest_ids = torch.cat(self.all_nearest_ids, 0).to(torch.int)
            self.render_path = torch.from_numpy(self.render_path).float()
        else:
            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_rays_real = self.all_rays
                self.all_rgbs = torch.cat(self.all_rgbs, 0)
                self.all_depths = torch.cat(self.all_depths, 0)
                self.all_dense_depths = torch.cat(self.all_dense_depths, 0)
                self.all_ids = torch.cat(self.all_ids, 0).to(torch.int)
                self.all_nearest_ids = torch.cat(self.all_nearest_ids, 0).to(torch.int)
            else:
                self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames']),h*w, 3)
                self.all_rays_real = self.all_rays
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames']),h,w,3)
                self.all_depths = torch.stack(self.all_depths, 0).reshape(-1,*self.img_wh[::-1], 1)
                self.all_dense_depths = torch.stack(self.all_dense_depths, 0).reshape(-1,*self.img_wh[::-1], 1)
                self.all_ids = torch.stack(self.all_ids, 0).to(torch.int)
                self.all_nearest_ids = torch.stack(self.all_nearest_ids, 0).to(torch.int)

                if self.split == 'test':
                    # check if the mask is in the mask folder
                    masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
                    # Get mask ids from filenames
                    masks_id = []
                    for mask in masks_lis:
                        # Get just the filename without path and extension
                        mask_name = os.path.basename(mask)
                        # Remove extension and try to convert to int
                        try:
                            mask_id = int(os.path.splitext(mask_name)[0])
                            masks_id.append(mask_id)
                        except ValueError:
                            print(f"Warning: Skipping mask file with invalid name format: {mask}")
                    
                    # Create full mask array initialized to True
                    masks_np = np.ones((len(self.frame_num), self.img_wh[1], self.img_wh[0], 1), dtype=bool)
                    
                    # For frames that have masks, load and resize them
                    for i, frame_id in enumerate(self.frame_num):
                        if frame_id in masks_id:
                            mask_path = [m for m in masks_lis if int(os.path.splitext(os.path.basename(m))[0]) == frame_id][0]
                            # Read mask and convert to grayscale if needed
                            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                            if mask is None:
                                print(f"Warning: Could not read mask file: {mask_path}")
                                continue
                            # Resize mask (note: cv2.resize expects (width, height))
                            mask = cv.resize(mask, (self.img_wh[0], self.img_wh[1]))
                            # Convert to binary and reshape to (H, W, 1)
                            masks_np[i] = (mask > 128)[..., None]
                            
                    self.all_masks = torch.from_numpy(masks_np).to(torch.bool)
                    self.all_masks = self.all_masks.reshape(-1, self.img_wh[1], self.img_wh[0], 1)
        
        f = (self.intrinsics[0][0, 0] + self.intrinsics[0][1, 1]) / 2
        self.focal = [self.intrinsics[0][0, 0], self.intrinsics[0][1, 1]]
        self.center = [self.intrinsics[0][0, 2], self.intrinsics[0][1, 2]]
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        print(f'{self.split} dataLoader loaded', len(self.all_rays), 'rays')
        
    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:
            sample = {'rays': self.all_rays[idx]}
        return sample