import os
import torch
import torch.backends.cudnn
import torch.utils.data
import numpy as np

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.logging

from time import time
import core.pipeline_train as pipeline
from core.test import test_net

from models.voxel_net.simple_3d_reconstruction import Simple3DReconstruction
from models.voxel_net.refiner import Refiner
from losses.losses import DiceLoss, CEDiceLoss, FocalLoss
from utils.average_meter import AverageMeter
from utils.scheduler_with_warmup import GradualWarmupScheduler

def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    # Load data
    train_data_loader, train_sampler, val_data_loader, val_file_num = pipeline.load_data(cfg)

    # Load models
    model = Simple3DReconstruction(cfg)
    refiner = Refiner(cfg)
    
    # Initialize training parameters
    init_epoch, best_iou, best_epoch, model, cfg = pipeline.setup_network(cfg, model)
    init_epoch, best_iou, best_epoch, refiner, cfg = pipeline.setup_refiner(cfg, refiner)    

    # Set up optimizers
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    refiner_optimizer = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    # Set up learning rate schedulers
    lr_scheduler = get_lr_scheduler(model_optimizer, cfg)
    refiner_lr_scheduler = get_lr_scheduler(refiner_optimizer, cfg)

    # Set up loss functions
    loss_function_model = get_loss_function(cfg.TRAIN.LOSS)
    loss_function_refiner = get_loss_function(cfg.TRAIN.LOSS)

    # Set up TensorBoard writers
    train_writer, val_writer = pipeline.setup_writer(cfg)
    
    # Training loop
    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
        epoch_start_time = time()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model_losses = AverageMeter()
        refiner_losses = AverageMeter()

        model.train()
        refiner.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
            rendering_images = rendering_images[:, :n_views_rendering, ...]
            data_time.update(time() - batch_end_time)
            
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
            
            # Forward pass through the main model
            output_3D = model(rendering_images)
            model_loss = loss_function_model(output_3D, ground_truth_volumes)

            model_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            model_optimizer.step()

            # Forward pass through the refiner using the model's output
            refined_output_3D = refiner(output_3D)
            refiner_loss = loss_function_refiner(refined_output_3D, ground_truth_volumes)

            refiner_optimizer.zero_grad()
            refiner_loss.backward()
            refiner_optimizer.step()

            # Logging losses
            model_losses.update(model_loss.item())
            refiner_losses.update(refiner_loss.item())

            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('Model/BatchLoss', model_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if batch_idx == 0 or (batch_idx + 1) % cfg.TRAIN.SHOW_TRAIN_STATE == 0:
                utils.logging.info(
                    f"[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}][Batch {batch_idx + 1}/{n_batches}] "
                    f"BatchTime = {batch_time.val:.3f} (s) DataTime = {data_time.val:.3f} (s) "
                    f"ModelLoss = {model_loss.item():.4f} RefinerLoss = {refiner_loss.item():.4f}"
                )
                print(f'LearningRate: {lr_scheduler.optimizer.param_groups[0]["lr"]:.6f}')
                print(f'Refiner LearningRate: {refiner_lr_scheduler.optimizer.param_groups[0]["lr"]:.6f}')
            else:
                utils.logging.debug(
                    f"[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}][Batch {batch_idx + 1}/{n_batches}] "
                    f"BatchTime = {batch_time.val:.3f} (s) DataTime = {data_time.val:.3f} (s) "
                    f"ModelLoss = {model_loss.item():.4f} RefinerLoss = {refiner_loss.item():.4f}"
                )

        torch.cuda.synchronize()

        lr_scheduler.step()
        refiner_lr_scheduler.step()

        train_writer.add_scalar('Model/EpochLoss', model_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        epoch_end_time = time()
        utils.logging.info(f"[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}] EpochTime = {epoch_end_time - epoch_start_time:.3f} (s) "
                           f"ModelLoss = {model_losses.avg:.4f} RefinerLoss = {refiner_losses.avg:.4f}")
            
        # Validate the model
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_file_num, val_writer, model, refiner)
        
        # Save weights if necessary
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            best_iou = max(best_iou, iou)
            best_epoch = epoch_idx if iou > best_iou else best_epoch
            save_checkpoint(cfg, epoch_idx, best_iou, best_epoch, model, refiner)

    train_writer.close()
    val_writer.close()

def get_lr_scheduler(optimizer, cfg):
    if cfg.TRAIN.LR_SCHEDULER == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR)
    elif cfg.TRAIN.LR_SCHEDULER == 'MilestonesLR':
        warm_up = 0 if cfg.TRAIN.RESUME_TRAIN else cfg.TRAIN.WARMUP
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[lr-warm_up for lr in cfg.TRAIN.MILESTONESLR.LR_MILESTONES],
            gamma=cfg.TRAIN.MILESTONESLR.GAMMA)
    else:
        raise ValueError(f'{cfg.TRAIN.LR_SCHEDULER} is not supported.')

def get_loss_function(loss_type):
    if loss_type == 1:
        return torch.nn.BCELoss()
    elif loss_type == 2:
        return DiceLoss()
    elif loss_type == 3:
        return CEDiceLoss()
    elif loss_type == 4:
        return FocalLoss()
    else:
        raise ValueError(f'Loss function type {loss_type} is not supported.')

def save_checkpoint(cfg, epoch_idx, best_iou, best_epoch, model, refiner):
    output_dir = cfg.DIR.CHECKPOINTS
    os.makedirs(output_dir, exist_ok=True)
    file_name = f'checkpoint-epoch-{epoch_idx:03d}.pth' if epoch_idx > best_epoch else 'checkpoint-best.pth'
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'refiner_state_dict': refiner.state_dict(),
    }
    torch.save(checkpoint, os.path.join(output_dir, file_name))
    utils.logging.info(f'Saved checkpoint to {os.path.join(output_dir, file_name)}')
