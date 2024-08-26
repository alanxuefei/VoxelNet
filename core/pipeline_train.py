#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Alan Xue <alan.xuefei@googlemail.com>

import os
import torch
import numpy as np
from utils import logging
import utils.data_loaders
import utils.data_transforms
import utils.helpers
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import tempfile

def modify_lr_strategy(cfg, current_epoch):
    if current_epoch <= cfg.TRAIN.WARMUP:
        raise ValueError('current_epoch <= cfg.TRAIN.WARM_UP; Please train from scratch!')
    
    if cfg.TRAIN.LR_SCHEDULER == 'ExponentialLR':
        init_lr = cfg.TRAIN.LEARNING_RATE
        current_epoch_lr = init_lr * (cfg.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR ** (current_epoch - cfg.TRAIN.WARMUP))
        cfg.TRAIN.LEARNING_RATE = current_epoch_lr
    elif cfg.TRAIN.LR_SCHEDULER == 'MilestonesLR':
        milestones = np.array(cfg.TRAIN.MILESTONESLR.LR_MILESTONES) - current_epoch
        init_lr = cfg.TRAIN.LEARNING_RATE
        current_epoch_lr = init_lr * cfg.TRAIN.MILESTONESLR.GAMMA ** len(np.where(milestones <= 0)[0])
        milestones = list(milestones[len(np.where(milestones <= 0)[0]):])
        cfg.TRAIN.MILESTONESLR.LR_MILESTONES = milestones
        cfg.TRAIN.LEARNING_RATE = current_epoch_lr
    else:
        raise ValueError(f'{cfg.TRAIN.LR_SCHEDULER} is not supported.')

    return cfg

def load_checkpoint_model(cfg, model):
    checkpoint_path = cfg.CHECKPOINT_FILE
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load the state_dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Retrieve saved training state
    init_epoch = checkpoint['epoch_idx']
    best_iou = checkpoint['best_iou']
    best_epoch = checkpoint['best_epoch']

    logging.info(f"Loaded model checkpoint from {checkpoint_path} (epoch {init_epoch}, best IOU {best_iou})")
    
    return init_epoch, best_iou, best_epoch

def load_checkpoint_refiner(cfg, refiner):
    checkpoint_path = cfg.CHECKPOINT_FILE
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load the state_dict into the refiner
    refiner.load_state_dict(checkpoint['refiner_state_dict'])

    # Retrieve saved training state
    init_epoch = checkpoint['epoch_idx']
    best_iou = checkpoint['best_iou']
    best_epoch = checkpoint['best_epoch']

    logging.info(f"Loaded refiner checkpoint from {checkpoint_path} (epoch {init_epoch}, best IOU {best_iou})")
    
    return init_epoch, best_iou, best_epoch

def load_data(cfg):
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])

    # Set up data loader
    train_dataset, _ = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms)
    val_dataset, val_file_num = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.CONST.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True)

    return train_data_loader, None, val_data_loader, val_file_num

def setup_network(cfg, model):
    # Set up network
    logging.info('Parameters in Model: %d.' % (utils.helpers.count_parameters(model)))

    # Move model to the appropriate device first
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize weights if starting from scratch
    model.apply(utils.helpers.init_weights)

    # Set sync batchnorm
    if cfg.TRAIN.SYNC_BN:
        logging.info('Setting sync_batchnorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        logging.info('Without sync_batchnorm')

    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    if cfg.RESUME_TRAIN:
        init_epoch, best_iou, best_epoch = load_checkpoint_model(cfg, model)
        cfg = modify_lr_strategy(cfg, init_epoch)

    return init_epoch, best_iou, best_epoch, model, cfg

def setup_refiner(cfg, refiner):
    # Set up refiner network
    logging.info('Parameters in Refiner: %d.' % (utils.helpers.count_parameters(refiner)))

    # Move refiner to the appropriate device first
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    refiner = refiner.to(device)

    # Initialize weights if starting from scratch
    refiner.apply(utils.helpers.init_weights)

    # Set sync batchnorm if applicable
    if cfg.TRAIN.SYNC_BN:
        logging.info('Setting sync_batchnorm for refiner ...')
        refiner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(refiner)
    else:
        logging.info('Without sync_batchnorm for refiner')

    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    if cfg.RESUME_TRAIN:
        init_epoch, best_iou, best_epoch = load_checkpoint_refiner(cfg, refiner)
        cfg = modify_lr_strategy(cfg, init_epoch)

    return init_epoch, best_iou, best_epoch, refiner, cfg

def setup_writer(cfg):
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    print(output_dir)
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
    return train_writer, val_writer

def solver(cfg, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    return optimizer
