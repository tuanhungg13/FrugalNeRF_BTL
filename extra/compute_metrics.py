import os
import numpy as np
from PIL import Image
import os
import torch
import configargparse
from skimage.metrics import structural_similarity
import cv2 as cv


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).to(torch.float32).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).to(torch.float32).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(gt_frame: np.ndarray, eval_frame: np.ndarray):
    """
    gt_frame: (H, W, 3)
    eval_frame: (H, W, 3)
    """
    assert gt_frame.shape == eval_frame.shape
    assert gt_frame.dtype == eval_frame.dtype

    return structural_similarity(gt_frame, eval_frame, channel_axis=-1, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False)

if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--render_dir", type=str, help="rendering directory")
    parser.add_argument("--gt_dir", type=str, help="ground truth directory")
    parser.add_argument("--mask_dir", type=str, default = None, help="mask directory")
    args = parser.parse_args()



    render_dir = args.render_dir
    gt_dir = args.gt_dir
    mask_dir = args.mask_dir
    
    outFile = f'{render_dir}/mean.txt'

    render_images = os.listdir(render_dir)
    render_images = [img for img in render_images if img.endswith('.png') or img.endswith('.jpg')]
    # remove the images that include depth
    render_images = [img for img in render_images if findItem(img.split('_'), 'depth') is None]
    gt_images = os.listdir(gt_dir)
    gt_images = [img for img in gt_images if img.endswith('.png') or img.endswith('.jpg')]
    if mask_dir is None:
        render_images.sort()
    else:
        render_images = sorted(render_images, key=lambda x: int(x.split('.')[0]))
    gt_images.sort()

    if gt_dir.find('llff') != -1:
        llff_hold = 8
        gt_images = gt_images[::llff_hold]
    if gt_dir.find('dtu') != -1:
        test_id = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11 , 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 45 , 46, 47]             
        mask_id = [1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47]
        if len(render_images) != len(mask_id):
            render_images_tmp = []
            for i in range(len(render_images)):
                if test_id[i] in mask_id:
                    render_images_tmp.append(render_images[i])
            render_images = render_images_tmp
        gt_images = [gt_images[i] for i in mask_id]
        mask_images = os.listdir(mask_dir)
        mask_images = [img for img in mask_images if img.endswith('.png') or img.endswith('.jpg')]
        mask_images.sort()
        if len(mask_images) != len(render_images):
            mask_images = [mask_images[i] for i in mask_id]
    if gt_dir.find('realestate10k') != -1:
        test_id = [5,6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35]
        gt_images = [gt_images[i] for i in test_id]

    with open(outFile, 'w') as f:
        all_psnr = []
        all_ssim = []
        all_alex = []
        for i in range(len(render_images)):
            render_img = Image.open(f'{render_dir}/{render_images[i]}')
            gt_img = Image.open(f'{gt_dir}/{gt_images[i]}')
            render_img = np.asarray(render_img, dtype=np.float32) / 255.0
            gt_img = np.asarray(gt_img, dtype=np.float32) / 255.0
            if mask_dir is not None:
                masks_np = (cv.resize(cv.imread(os.path.join(mask_dir, mask_images[i])),(400,300))>128).astype(np.float32)
                render_img = render_img * masks_np #+ (1-masks_np)
                gt_img = gt_img * masks_np #+ (1-masks_np)

            psnr = -10. * np.log(np.mean(np.square(render_img - gt_img))) / np.log(10.)
            ssim = rgb_ssim(render_img, gt_img)
            lpips_alex = rgb_lpips(gt_img, render_img, 'alex','cuda')
            print(f'{render_images[i]} : psnr {psnr} ssim {ssim}  l_a {lpips_alex}')
            

            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_alex.append(lpips_alex)

        psnr = np.mean(np.array(all_psnr))
        ssim = np.mean(np.array(all_ssim))
        l_a  = np.mean(np.array(all_alex))

        print(f'mean : psnr {psnr} ssim {ssim}  l_a {l_a}')
        f.write(f'{psnr}\n{ssim}\n{l_a}\n')