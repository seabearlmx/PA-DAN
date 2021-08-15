import os
import sys
from pathlib import Path
import math
from numpy import inf
import random
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from padan.model.discriminator import get_fc_discriminator
from padan.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from padan.utils.func import loss_calc, bce_loss
from padan.utils.loss import entropy_loss
from padan.utils.func import prob_2_entropy
from padan.utils.viz_segmask import colorize_mask

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

checkpoint = 0
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )


def train_dada(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # d_perceptual = FCDiscriminator_low(2048)
    d_perceptual = get_fc_discriminator(num_classes=num_classes)
    d_perceptual.train()
    d_perceptual.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer.zero_grad()

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
    optimizer_d_main.zero_grad()

    optimizer_d_perceptual = optim.Adam(d_perceptual.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
    optimizer_d_perceptual.zero_grad()

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        i_iters = i_iter + checkpoint

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_d_perceptual.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iters, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iters, cfg)
        adjust_learning_rate_discriminator(optimizer_d_perceptual, i_iters, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        for param in d_perceptual.parameters():
            param.requires_grad = False

        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch

        _, batch = targetloader_iter.__next__()
        images, trg_labels, _, _ = batch

        _, pred_src_perceptual_seg, pred_src_main = model(images_source.to(device))

        pred_src_main = interp_target(pred_src_main)
        pred_src_perceptual_seg = interp_target(pred_src_perceptual_seg)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss_seg_src_perceptual = loss_calc(pred_src_perceptual_seg, labels, device)

        loss = loss_seg_src_main + loss_seg_src_perceptual
        loss.backward()

        # adversarial training ot fool the discriminator
        _, pred_trg_perceptual_seg, pred_trg_main = model(images.to(device))

        pred_trg_main = interp_target(pred_trg_main)
        # loss_seg_trg_main = loss_calc(pred_trg_main, trg_labels, device)
        pred_trg_perceptual_seg = interp_target(pred_trg_perceptual_seg)
        # loss_seg_trg_perceptual = loss_calc(pred_trg_perceptual_seg, trg_labels, device)

        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)

        d_out_perceptual = d_perceptual(prob_2_entropy(F.softmax(pred_trg_perceptual_seg, dim=1)))
        loss_adv_trg_perceptual = bce_loss(d_out_perceptual, source_label)

        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main \
               + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_perceptual
               # + cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_trg_main \
               # + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_trg_perceptual
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_main.parameters():
            param.requires_grad = True
        for param in d_perceptual.parameters():
            param.requires_grad = True

        # train with source
        pred_src_main = pred_src_main.detach()
        pred_src_perceptual_seg = pred_src_perceptual_seg.detach()

        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2

        d_out_perceptual = d_perceptual(prob_2_entropy(F.softmax(pred_src_perceptual_seg, dim=1)))
        loss_d_perceptual = bce_loss(d_out_perceptual, source_label)
        loss_d_perceptual = loss_d_perceptual / 2

        loss_d_main.backward()
        loss_d_perceptual.backward()

        # train with target
        pred_trg_main = pred_trg_main.detach()
        pred_trg_perceptual_seg = pred_trg_perceptual_seg.detach()

        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2

        d_out_perceptual = d_perceptual(prob_2_entropy(F.softmax(pred_trg_perceptual_seg, dim=1)))
        loss_d_perceptual = bce_loss(d_out_perceptual, target_label)
        loss_d_perceptual = loss_d_perceptual / 2

        loss_d_main.backward()
        loss_d_perceptual.backward()

        optimizer.step()
        optimizer_d_main.step()
        optimizer_d_perceptual.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          'loss_seg_src_perceptual':loss_seg_src_perceptual,
                          # 'loss_seg_trg_main': loss_seg_trg_main,
                          # 'loss_seg_trg_perceptual': loss_seg_trg_perceptual,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_adv_trg_perceptual': loss_adv_trg_perceptual,
                          'loss_d_perceptual': loss_d_perceptual,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iters)

        if i_iters % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iters != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iters}.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iters}_D_main.pth')
            if i_iters >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()


def print_losses(current_losses, i_iters):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iters} {full_string}')


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation_with_padan(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'PADAN':
        train_dada(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
