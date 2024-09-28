#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Alan Xue <alan.xuefei@googlemail.com>

import os
import torch
from utils import logging
import utils.data_loaders
import utils.data_transforms
import utils.helpers

def save_checkpoint(cfg, epoch_idx, iou, model, refiner, model_loss, refiner_loss):
    output_dir = cfg.DIR.OUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    # Extract additional information
    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    embed_dim = cfg.NETWORK.EMBED_DIM
    attention_heads = cfg.NETWORK.ATTENTION_HEADS

    # Format best_iou, model_loss, and refiner_loss for filename by replacing '.' with '_'
    best_iou_str = f'{iou:.4f}'.replace('.', '_')
    model_loss_str = f'{model_loss:.4f}'.replace('.', '_')
    refiner_loss_str = f'{refiner_loss:.4f}'.replace('.', '_')

    # Include model_loss and refiner_loss in the file name
    file_name = (f'checkpoint-epoch-{epoch_idx:03d}-views{n_views_rendering}-'
                 f'embed{embed_dim}-heads{attention_heads}-iou{best_iou_str}-'
                 f'model_loss{model_loss_str}-refiner_loss{refiner_loss_str}.pth')

    file_path = os.path.join(output_dir, file_name)

    # Create the checkpoint dictionary
    checkpoint = {
        'cfg': cfg,  # Include the full configuration object
        'epoch_idx': epoch_idx,
        'best_iou': iou,
        'model_loss': model_loss,
        'refiner_loss': refiner_loss,
        'model_state_dict': model.state_dict(),
        'refiner_state_dict': refiner.state_dict(),
    }

    # Save the checkpoint
    torch.save(checkpoint, file_path)
    logging.info(f'Saved checkpoint to {file_path}')

def load_checkpoint_model(cfg, model):
    checkpoint_path = cfg.CHECKPOINT_MODEL_FILE

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    # Get the model's current state_dict
    model_state_dict = model.state_dict()

    # Create a new state_dict to load, filtering out mismatched keys
    filtered_state_dict = {}
    mismatched_keys = []
    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.size() == model_state_dict[name].size():
                filtered_state_dict[name] = param
            else:
                mismatched_keys.append(name)
                logging.info(f"Size mismatch for '{name}': "
                                f"checkpoint param of size {param.size()} "
                                f"does not match model param of size {model_state_dict[name].size()}, skipping.")
        else:
            logging.info(f"'{name}' not found in the current model, skipping.")

    # Load the filtered state_dict
    model.load_state_dict(filtered_state_dict, strict=False)

    # Retrieve saved training state
    init_epoch = checkpoint.get('epoch_idx', 0)
    best_iou = checkpoint.get('best_iou', -1)
    best_epoch = checkpoint.get('best_epoch', -1)

    logging.info(f"Loaded model checkpoint from {checkpoint_path} "
                 f"(epoch {init_epoch}, best IoU {best_iou})")
    logging.info(f"Total parameters loaded: {len(filtered_state_dict)}")
    logging.info(f"Total parameters skipped due to mismatch: {len(mismatched_keys)}")

    return init_epoch, best_iou, best_epoch

def load_checkpoint_refiner(cfg, refiner):
    checkpoint_path = cfg.CHECKPOINT_REFINER_FILE
    
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

    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    if cfg.RESUME_TRAIN:
        init_epoch, best_iou, best_epoch = load_checkpoint_model(cfg, model)

    return init_epoch, best_iou, best_epoch, model, cfg

def setup_refiner(cfg, refiner):
    # Set up refiner network
    logging.info('Parameters in Refiner: %d.' % (utils.helpers.count_parameters(refiner)))

    # Move refiner to the appropriate device first
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    refiner = refiner.to(device)

    # Initialize weights if starting from scratch
    refiner.apply(utils.helpers.init_weights)

    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    if cfg.RESUME_TRAIN:
        init_epoch, best_iou, best_epoch = load_checkpoint_refiner(cfg, refiner)
    return init_epoch, best_iou, best_epoch, refiner, cfg

def solver(cfg, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    return optimizer
