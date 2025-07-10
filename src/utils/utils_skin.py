import os
import torch

import pandas as pd
import matplotlib.pyplot as plt
from medpy.metric import dc, hd95
import datetime
import cv2
import numpy as np
from PIL import Image


def calc_iou(pred, gt):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    
    Args:
        pred (np.ndarray): Predicted binary mask.
        gt (np.ndarray): Ground truth binary mask.
        
    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
    return iou_score


def histogram_equalization_rgb(image: np.ndarray) -> np.ndarray:
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Equalize the Y channel
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    
    # Convert back to RGB
    equalized_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return equalized_img



def save_im_gt_pd_hot(im, gt, pd, label, save_path="../results/vis/skin"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if isinstance(im, torch.Tensor):
        im = im[0, :].detach().cpu().numpy().transpose(1, 2, 0)
        gt = gt[0, 0].detach().cpu().numpy()
        pd = pd[0, 1].detach().cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(im[:, :, :3])
    plt.title("Image")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(gt, cmap="jet")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(pd, cmap="jet")
    plt.title("Prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{label}.jpeg"))
    plt.close()


def skin_plot(img, gt, pred):
    edged_test = cv2.Canny(pred, 100, 255)
    contours_test, _ = cv2.findContours(edged_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edged_gt = cv2.Canny(gt, 100, 255)
    contours_gt, _ = cv2.findContours(edged_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_test in contours_test:
        cv2.drawContours(img, [cnt_test], -1, (0, 0, 255), 1)
    for cnt_gt in contours_gt:
        cv2.drawContours(img, [cnt_gt], -1, (0,255,0), 1)
    return img

def save_im_gt_pd(im, gt, pd, fid, save_path="../results/vis/skin"):
        os.makedirs(save_path, exist_ok=True)
        im = (im - im.min()) / (im.max() - im.min())
        gt = (gt - gt.min()) / (gt.max() - gt.min())
        pd = (pd - pd.min()) / (pd.max() - pd.min())
        im = np.ascontiguousarray(im*255., dtype=np.uint8)
        img = im.copy()
        gt = np.uint8(gt*255.)
        pd = np.ascontiguousarray(pd*255., dtype=np.uint8)
        res_img = skin_plot(im, gt, pd)
        
        Image.fromarray(img).save(f"{save_path}/{fid}_img.png")
        Image.fromarray(gt).save(f"{save_path}/{fid}_gt.png")
        Image.fromarray(res_img).save(f"{save_path}/{fid}_img_gt_pred.png")



def val(net, vl_loader, logging, best_dcs, epoch=0):
    logging.info("Validation ===>")
    dc_sum = 0
    net.eval()
    for i, val_sampled_batch in enumerate(vl_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()
        val_outputs = net(val_image_batch)
        val_outputs_binary = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        dc_sum += dc(val_outputs_binary.detach().cpu().numpy(), val_label_batch[:].detach().cpu().numpy())
    performance = dc_sum / len(vl_loader)

    # print(f"Saving vis. results for validation at epoch: {epoch:03d}")
    save_im_gt_pd_hot(val_image_batch, val_label_batch, val_outputs, f"{epoch:04d}")

    logging.info('performance in val model) mean_dice:%f, best_dice:%f' % (performance, best_dcs))
    return performance


# def test(net, te_loader, logging, best_dcs):
#     logging.info("Test ===>")
#     dc_sum = 0
#     net.eval()
#     for i, val_sampled_batch in enumerate(te_loader):
#         val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
#         val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
#         val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()
#         val_outputs = net(val_image_batch)
#         val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
#         dc_sum += dc(val_outputs.detach().cpu().numpy(), val_label_batch[:].detach().cpu().numpy())
#     performance = dc_sum / len(te_loader)
#     logging.info('performance in test model) mean_dice:%f, best_dice:%f' % (performance, best_dcs))
#     return performance

def test(net, te_loader, logging, best_dcs, save_path=None):
    logging.info("Test ===>")
    dc_sum = 0
    acc_sum = 0
    total_pixels = 0
    ious = []
    net.eval()
    for i, val_sampled_batch in enumerate(te_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_id = val_sampled_batch["id"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()
        val_outputs = net(val_image_batch)
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

        pd = val_outputs.detach().cpu().numpy()
        gt = val_label_batch[0, 0].detach().cpu().numpy()
        # Accuracy calculation
        correct = (pd == gt).sum()
        acc_sum += correct
        total_pixels += gt.size

        if save_path is not None:
            # Save the image, ground truth, and prediction
            im = val_image_batch[0, :3].cpu().detach().numpy().transpose(1, 2, 0)
            id = val_id[0]
            save_im_gt_pd(im, gt, pd, id.item(), save_path=save_path)
        ious.append(calc_iou(pd>0.5, gt>0.5))

        dc_sum += dc(pd, val_label_batch[:].detach().cpu().numpy())
        # dc_sum += dc(val_outputs.detach().cpu().numpy(), val_label_batch[:].detach().cpu().numpy())

    avg_dice = dc_sum / len(te_loader)
    avg_iou = np.mean(ious)
    avg_acc = acc_sum / total_pixels
    if best_dcs < 0 or best_dcs is None:
        print('performance in test model) mean_dice:%f, iou:%f, acc:%f' % (avg_dice, avg_iou, avg_acc))
    else:
        print('performance in test model) mean_dice:%f, best_dice:%f, iou:%f, acc:%f' % (avg_dice, best_dcs, avg_iou, avg_acc))
    return avg_dice, avg_acc, avg_iou


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')
