import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms
import logging


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

from matplotlib import pyplot as plt
import os
def save_im_gt_pd(im, gt, pd, label, save_path="../results/vis/synapse"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    idx = im.shape[0] // 2
    im = im[idx]
    gt = gt[idx]
    pd = pd[idx]
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(im, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(gt)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(pd)
    plt.title("Prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"te_{label}.jpg"))
    plt.close()

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1. ,epoch=0):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input = x_transforms(slice).unsqueeze(0).float().cuda()
            with torch.no_grad():
                outputs = net(input)
                # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    save_im_gt_pd(image, label, prediction, f"{epoch:04d}_{case}")

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

# Written By Sina 
class AddNoise(object):
    def __init__(self, noise_type='none', **kwargs):
        self.noise_type = noise_type
        self.params = kwargs
    
    def set_params(self, **kwargs):
        self.params = kwargs

    def __call__(self, sample):
        if self.noise_type == 'salt_pepper':
            return self.add_salt_pepper_noise(sample)
        elif self.noise_type == 'gaussian':
            return self.add_gaussian_noise(sample)
        elif self.noise_type == 'poisson':
            return self.add_poisson_noise(sample)
        elif self.noise_type == 'speckle':
            return self.add_speckle_noise(sample)
        else:
            return sample
    
    def add_salt_pepper_noise(self, sample):
        salt_prob = self.params.get('salt_prob', 0.05)
        pepper_prob = self.params.get('pepper_prob', 0.05)
        salt_mask = torch.rand(sample.shape) < salt_prob
        pepper_mask = torch.rand(sample.shape) < pepper_prob
        noisy_sample = sample.clone()
        noisy_sample[salt_mask] = 1
        noisy_sample[pepper_mask] = 0
        return noisy_sample
    
    def add_gaussian_noise(self, sample):
        mean = self.params.get('mean', 0)
        std = self.params.get('std', 0.1)
        noisy_sample = sample + torch.randn_like(sample) * std + mean
        return noisy_sample
    
    def add_poisson_noise(self, sample):
        sample = sample.float()
        vals = 2 ** torch.ceil(torch.log2(torch.tensor([len(torch.unique(sample))]))).item()
        noisy_sample = torch.poisson(sample * vals) / vals
        return noisy_sample
    
    def add_speckle_noise(self, sample):
        std = self.params.get('std', 0.1)
        gaussian_noise = torch.randn_like(sample) * std
        noisy_sample = sample + sample * gaussian_noise
        return noisy_sample
    
    
    
def inference(model, te_loader, args, test_save_path=None, epoch=0):
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(te_loader):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # if not "case0003" in case_name: continue
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, epoch=epoch)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(te_loader.dataset)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d -> mean_dice: %f, mean_hd95: %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(f'Testing performance in best val model) mean_dice: {performance}, mean_hd95: {mean_hd95}')
    return performance, mean_hd95



