import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda', reso=1):

    rgbs, depth_maps, weights, sigmas, m_list, rgb_rays = [], [], [], [], [], []
    rgb_maps_MR, depth_maps_MR = [], []
    rgb_maps_LR, depth_maps_LR = [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        if is_train:
            rgb_map, depth_map, weight, m, sigma, rgb_ray = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, reso=reso)
            weights.append(weight)
            m_list.append(m)
            sigmas.append(sigma)
            rgb_rays.append(rgb_ray)
        else:
            rgb_map, depth_map, _, _, _, _ = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, reso=1)
            rgb_map_MR, depth_map_MR, _, _, _, _ = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, reso=4)
            rgb_map_LR, depth_map_LR, _, _, _, _ = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, reso=16)
            rgb_maps_MR.append(rgb_map_MR)
            depth_maps_MR.append(depth_map_MR)
            rgb_maps_LR.append(rgb_map_LR)
            depth_maps_LR.append(depth_map_LR)
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)   
        
    if is_train:
        return torch.cat(rgbs), torch.cat(depth_maps), torch.cat(weights), torch.cat(m_list), torch.cat(sigmas), torch.cat(rgb_rays)
    else:
        return torch.cat(rgbs), torch.cat(depth_maps), torch.cat(rgb_maps_MR), torch.cat(depth_maps_MR), torch.cat(rgb_maps_LR), torch.cat(depth_maps_LR)

@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "/histograms", exist_ok=True)  # Directory to save histograms

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        rgb_map, depth_map, rgb_map_MR, depth_map_MR, rgb_map_LR, depth_map_LR = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_LR = rgb_map_LR.clamp(0.0, 1.0)
        rgb_map_MR = rgb_map_MR.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        rgb_map_MR, depth_map_MR = rgb_map_MR.reshape(H, W, 3).cpu(), depth_map_MR.reshape(H, W).cpu()
        rgb_map_LR, depth_map_LR = rgb_map_LR.reshape(H, W, 3).cpu(), depth_map_LR.reshape(H, W).cpu()

        depth_map_np = depth_map.numpy()
        depth_map_MR_np = depth_map_MR.numpy()
        depth_map_LR_np = depth_map_LR.numpy()

        # Convert depth maps for RGBD concatenation
        depth_map, _ = visualize_depth_numpy(depth_map_np, None)
        depth_map_MR, _ = visualize_depth_numpy(depth_map_MR_np, None)
        depth_map_LR, _ = visualize_depth_numpy(depth_map_LR_np, None)
        
        # Continue with the existing PSNR, SSIM, and image saving logic...
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)

            if test_dataset.all_masks is not None:
                mask = test_dataset.all_masks[idxs[idx]].view(H, W, 1)
                rgb_map = rgb_map * mask
                gt_rgb = gt_rgb * mask

            
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map.numpy(), gt_rgb.numpy())
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_map_MR = (rgb_map_MR.numpy() * 255).astype('uint8')
        rgb_map_LR = (rgb_map_LR.numpy() * 255).astype('uint8')
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map, rgb_map_MR, depth_map_MR, rgb_map_LR, depth_map_LR), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx}.png', rgb_map)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    if compute_extra_metrics:
        return PSNRs, ssims, l_alex
    else:
        return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps, depth_maps_MR, depth_maps_LR = [], [], [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        if test_dataset.dataset_name == "dtu":
            rays_o, rays_d = test_dataset.gen_rays_at(test_dataset.intrinsics.mean(0), c2w)
        else:
            rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, depth_map, _, depth_map_MR, _, depth_map_LR = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        depth_map_MR = depth_map_MR.reshape(H, W).cpu()
        depth_map_LR = depth_map_LR.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),None)
        depth_map_MR, _ = visualize_depth_numpy(depth_map_MR.numpy(),None)
        depth_map_LR, _ = visualize_depth_numpy(depth_map_LR.numpy(),None)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        depth_maps_MR.append(depth_map_MR)
        depth_maps_LR.append(depth_map_LR)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map, depth_map_MR, depth_map_LR), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=4)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=4)
    imageio.mimwrite(f'{savePath}/{prtx}depthMRvideo.mp4', np.stack(depth_maps_MR), fps=30, quality=4)
    imageio.mimwrite(f'{savePath}/{prtx}depthLRvideo.mp4', np.stack(depth_maps_LR), fps=30, quality=4)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

