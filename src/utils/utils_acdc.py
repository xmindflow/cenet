import torch
import numpy as np
import logging
from utils import test_single_volume


def inference(args, model, testloader, test_save_path=None, epoch=0):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, epoch=epoch)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
            np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        #mean_jacard = np.mean(metric_list, axis=0)[2]
        #mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info(
            'Testing performance in best val model) mean_dice:%f mean_hd95:%f' % (performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance, mean_hd95
