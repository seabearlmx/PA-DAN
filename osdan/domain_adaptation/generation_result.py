import os.path as osp
import os
import time
import scipy
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from osdan.utils.func import per_class_iu, fast_hist
from osdan.utils.serialization import pickle_dump, pickle_load
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

output_save_path = './result/conv5_output'
osdan_save_path = './result/osdan_output'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    i_iter = None
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
    print("Evaluating model", restore_from)
    load_checkpoint_for_evaluation(models[0], restore_from, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    osdan_hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = iter(test_loader)
    for index in tqdm(range(len(test_loader))):
        if index % 100 == 0:
            print('%d processed' % index)
        image, label, _, name = next(test_iter)
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            _, pred_osdan, pred_main = models[0](image.cuda(device))
            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
            output_save = np.asarray(output, dtype=np.uint8)
            output_col = colorize_mask(output_save)
            output_save = Image.fromarray(output_save)

            osdan = interp(pred_osdan).cpu().data[0].numpy()
            osdan = osdan.transpose(1, 2, 0)
            osdan = np.argmax(osdan, axis=2)
            osdan_save = np.asarray(osdan, dtype=np.uint8)
            osdan_col = colorize_mask(osdan_save)
            osdan_save = Image.fromarray(osdan_save)

            name = name[0].split('/')[-1]
            output_save.save('%s/%s' % (output_save_path, name))
            output_col.save('%s/%s_color.png' % (output_save_path, name.split('.')[0]))
            osdan_save.save('%s/%s' % (osdan_save_path, name))
            osdan_col.save('%s/%s_color.png' % (osdan_save_path, name.split('.')[0]))

        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        osdan_hist += fast_hist(label.flatten(), osdan.flatten(), cfg.NUM_CLASSES)
        if verbose and index > 0 and index % 100 == 0:
            print('output : {:d} / {:d}: {:0.2f}'.format(
                index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            print('osdan: {:d} / {:d}: {:0.2f}'.format(
                index, len(test_loader), 100 * np.nanmean(per_class_iu(osdan_hist))))
    inters_over_union_classes = per_class_iu(hist)
    osdan_inters_over_union_classes = per_class_iu(osdan_hist)
    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    osdan_computed_miou = round(np.nanmean(osdan_inters_over_union_classes) * 100, 2)
    print('\tOutput mIoU:', computed_miou)
    print('\tOsdan model:', osdan_computed_miou)
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)
        display_stats(cfg, test_loader.dataset.class_names, osdan_inters_over_union_classes)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    # , map_location={'cuda:3':'cuda:0'}
    print(checkpoint)
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))


if __name__ == '__main__':
    main()