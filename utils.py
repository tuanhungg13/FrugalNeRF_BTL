import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal
from skimage.metrics import structural_similarity
from torchmetrics.functional.regression import pearson_corrcoef
from torch_efficient_distloss import eff_distloss

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
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

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None

def rgb_ssim(gt_frame: np.ndarray, eval_frame: np.ndarray):
    """
    gt_frame: (H, W, 3)
    eval_frame: (H, W, 3)
    """
    assert gt_frame.shape == eval_frame.shape
    assert gt_frame.dtype == eval_frame.dtype

    return structural_similarity(gt_frame, eval_frame, channel_axis=-1, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False)


import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        count_w = max(count_w, 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class colorTVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(colorTVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x,y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv_x = torch.mean(-torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]),dim=1,keepdim=True)
        w_tv_x = torch.mean(-torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]),dim=1,keepdim=True)
        weight_h_x = torch.exp(h_tv_x)
        weight_w_x = torch.exp(w_tv_x)
        
        h_y = y.size()[2]
        w_y = y.size()[3]
        count_h = self._tensor_size(y[:,:,1:,:])
        count_w = self._tensor_size(y[:,:,:,1:])
        count_w = max(count_w, 1)
        h_tv_y = torch.pow((y[:,:,1:,:]-y[:,:,:h_y-1,:])*weight_h_x,2).sum()
        w_tv_y = torch.pow((y[:,:,:,1:]-y[:,:,:,:w_y-1])*weight_w_x,2).sum()

        return self.TVLoss_weight*2*(h_tv_y/count_h+w_tv_y/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def depth_tv_loss(depth_map_patch):
    d_x = depth_map_patch[:,1:,:] - depth_map_patch[:,:-1,:]
    d_y = depth_map_patch[1:,:,:] - depth_map_patch[:-1,:,:]
    d_x = torch.mean(torch.pow(d_x,2))
    d_y = torch.mean(torch.pow(d_y,2))
    return d_x + d_y

def cal_disparity_loss(pre_depth, gt_depth, weight=None):
    if weight is None: 
        return torch.mean((1.0/(pre_depth+1e-6) - 1.0/(gt_depth+1e-6))**2)
    else:
        return torch.mean(weight * (1.0/(pre_depth+1e-6) - 1.0/(gt_depth+1e-6))**2)
    
def depth_l2_loss(rendered_depth, gt_depth, weight=None):
    if weight is None: 
        return torch.mean((rendered_depth - gt_depth)**2)
    else:
        return torch.mean(weight * (rendered_depth - gt_depth)**2)

def scale_invariant_depth_loss(rendered_depth, gt_depth):
    return min((1 - pearson_corrcoef( - rendered_depth, gt_depth)), (1 - pearson_corrcoef(1 / (rendered_depth + 200.),  gt_depth)))

def cal_occ_loss(sigma, rgb_ray, reg_range, wb_range, wb_prior=False):
    rgb_mean = rgb_ray.mean(-1)
    # Compute a mask for the white/black background region if using a prior
    if wb_prior:
        white_mask = rgb_mean > 0.99 # A naive way to locate white background
        black_mask = rgb_mean < 0.01  # A naive way to locate black background
        rgb_mask = (white_mask | black_mask) # White or black background
        rgb_mask[:, wb_range:] = 0 # White or black background range
    else:
        rgb_mask = torch.zeros_like(rgb_mean)
    
    # Create a mask for the general regularization region
    # It can be implemented as a one-line-code.
    if reg_range > 0:
        rgb_mask[:, :reg_range] = 1# Penalize the points in reg_range close to the camera

    # Compute the density-weighted loss within the regularization and white/black background mask
    return torch.mean(sigma * rgb_mask)

def distortion_loss(weight, m, nSamples):
    return eff_distloss(weight, m.detach(), 1/nSamples)

import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


@torch.no_grad()
def warping(allrays, ray_idx, H, W, focal, depth_map, c2w_j, startposition_in_allray, patch_mask, patch_size, device):
    depth_map_patch = torch.cat([depth_map]*(patch_size**2))
    c2w_j_patch = torch.cat([c2w_j] * (patch_size**2))
    startposition_in_allray_patch = torch.cat([startposition_in_allray] * (patch_size**2)).to(device)
    
    
    rays_train = torch.zeros((ray_idx.shape[0], 6), device=device)
    rays_train[patch_mask] = allrays[ray_idx[patch_mask].cpu()].to(device)
    # camera coordiante i to camera coordinate j
    xyz = rays_train[:, 0:3] + depth_map_patch.unsqueeze(-1) * rays_train[:, 3:]
    intrinsic = torch.tensor([[focal[0], 0, W/2], [0, focal[1], H/2], [0, 0, 1]], dtype=rays_train.dtype).to(device)
    intrinsic = intrinsic.unsqueeze(0)
    pos_2 = reproject(xyz, c2w_j_patch, intrinsic).round().long()
    u = pos_2[:, 0]
    v = pos_2[:, 1]
    within_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    position_in_allray = startposition_in_allray_patch.to(device) + v * W + u
    return position_in_allray, within_mask


def reproject(points_to_reproject, poses_to_reproject_to, intrinsics,):
    """

    Args:
        points_to_reproject: (num_rays, )
        poses_to_reproject_to: (num_poses, 4, 4)
        intrinsics: (num_poses, 3, 3)

    Returns:

    """
    other_views_origins = poses_to_reproject_to[:, :3, 3]
    other_views_rotations = poses_to_reproject_to[:, :3, :3]
    reprojected_rays_d = points_to_reproject - other_views_origins

    # for changing coordinate system conventions
    permuter = torch.eye(3).to(points_to_reproject.device)
    permuter[1:] *= -1
    intrinsics = intrinsics[:1]

    pos_2 = (intrinsics @ permuter[None] @ other_views_rotations.transpose(1, 2) @ reprojected_rays_d[..., None]).squeeze()
    pos_2 = pos_2[:, :2] / pos_2[:, 2:]
    return pos_2


@ torch.no_grad()
def cal_reprojection_error(rgb, projected_rgb, mask,  patch_size):    
    repro_error = torch.ones(mask.shape[0]).to(mask.device)
    repro_error[mask] = torch.mean((rgb - projected_rgb)**2, -1)
    repro_error = torch.sqrt(torch.mean(repro_error.view(int(repro_error.shape[0] / (patch_size**2)), patch_size**2), dim=-1)) # get the mean reprojection error of each patch
    return repro_error


# calculate reprojection error with rgb of frame i and rgb warped to frame j
@torch.no_grad()
def patchify(ray_idx, H, W, patch_size, total_frame_len, device):
    patch_offset = patch_size // 2
    t_ref = (ray_idx // (H * W)).to(device).unsqueeze(-1).repeat(1, patch_size**2)  # frame num
    v_ref = ((ray_idx % (H * W)) // W).to(device).unsqueeze(-1).repeat(1, patch_size**2) + torch.tensor([i - patch_offset for i in range(patch_size)], device=device).repeat(patch_size)
    u_ref = ((ray_idx % (H * W)) % W).to(device).unsqueeze(-1).repeat(1, patch_size**2) + torch.tensor([j - patch_offset for j in range(patch_size)], device=device).repeat_interleave(patch_size)
    patch_ray_idx = t_ref * (H * W) + v_ref * W + u_ref
    patch_ray_idx = patch_ray_idx.view(-1)  # (batch_size * k * k)
    patch_mask = ((u_ref >= 0) & (u_ref < W) & (v_ref >= 0) & (v_ref < H)) & (t_ref >= 0) & (t_ref < total_frame_len)
    patch_mask = patch_mask.view(-1)  # mask for pixels out of (H, W)

    return patch_ray_idx, patch_mask

class DepthEstimator:
    def __init__(self, downsampling=1):
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        for param in self.midas.parameters():
            param.requires_grad = False
            
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        self.downsampling = downsampling
        
    def estimate_depth(self, img, mode='test'):
        h, w = img.shape[1:3]
        norm_img = (img[None] - 0.5) / 0.5
        norm_img = torch.nn.functional.interpolate(
            norm_img,
            size=(384, 512),
            mode="bicubic",
            align_corners=False)

        if mode == 'test':
            with torch.no_grad():
                prediction = self.midas(norm_img)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h//self.downsampling, w//self.downsampling),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
        else:
            prediction = self.midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//self.downsampling, w//self.downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction
