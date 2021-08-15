import os.path as osp
import os
import time
import scipy
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import logging

from padan.utils.func import per_class_iu, fast_hist
from padan.utils.serialization import pickle_dump, pickle_load
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

output_save_path = './target_ssl'

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
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    i_iter = None
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
    print("Evaluating model", restore_from)
    load_checkpoint_for_evaluation(models[0], restore_from, device)
    # eval
    predicted_label = np.zeros((len(test_loader), 512, 1024))
    predicted_prob = np.zeros((len(test_loader), 512, 1024))
    image_name = []
    test_iter = iter(test_loader)
    for index in tqdm(range(len(test_loader))):
        if index % 100 == 0:
            print('%d processed' % index)
        image, _, _, name = next(test_iter)
        with torch.no_grad():
            pred_main = models[0](image.cuda(device))[1]
        output = nn.functional.softmax(pred_main, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])

    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x) * 0.5))])
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    for index in range(len(test_loader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob < thres[i]) * (label == i)] = 255
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (output_save_path, name))


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


if __name__ == '__main__':
    main()