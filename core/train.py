import os
import time
import torch
import torch.backends.cudnn
import torch.utils.data
import numpy as np

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.logging

import core.pipeline_train as pipeline
from core.test import test_net, test_net_fscore

from models.voxel_net.voxel_net_t import voxelNet
from models.voxel_net.refiner import Refiner, DummyRefiner
from losses.losses import get_loss_function
from utils.average_meter import AverageMeter

def train_model(cfg, model, refiner, train_data_loader, model_optimizer, loss_function_model, epoch_idx):
    """Train the model for one epoch, computing both model and refiner losses."""
    model.train()
    refiner.eval()  # Ensure the refiner is in evaluation mode

    # Freeze refiner parameters
    for param in refiner.parameters():
        param.requires_grad = False

    # Initialize meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model_losses = AverageMeter()
    refiner_losses = AverageMeter()  # Added to track refiner loss

    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    batch_end_time = time.time()
    n_batches = len(train_data_loader)

    for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
        rendering_images = rendering_images[:, :n_views_rendering, ...]
        data_time.update(time.time() - batch_end_time)
        
        rendering_images = utils.helpers.var_or_cuda(rendering_images)
        ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

        # Forward pass through the model
        output_3D, _ = model(rendering_images)
        # Compute model loss
        model_loss = loss_function_model(output_3D, ground_truth_volumes)
        model_losses.update(model_loss.item())

        # Backward pass and optimization
        model_optimizer.zero_grad()
        model_loss.backward()
        model_optimizer.step()

        # Forward pass through the refiner (without gradients)
        with torch.no_grad():
            refined_output_3D = refiner(output_3D)
            # Compute refiner loss
            refiner_loss = loss_function_model(refined_output_3D, ground_truth_volumes)
            refiner_losses.update(refiner_loss.item())

        batch_time.update(time.time() - batch_end_time)
        batch_end_time = time.time()

        # Logging
        if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
            utils.logging.info(
                f"[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}]"
                f"[Batch {batch_idx + 1}/{n_batches}] "
                f"BatchTime = {batch_time.val:.3f} (s) "
                f"DataTime = {data_time.val:.3f} (s) "
                f"ModelLoss = {model_losses.val:.4f} "
                f"RefinerLoss = {refiner_losses.val:.4f}"
            )

    # Optionally, you can return both average losses
    return model_losses.avg, refiner_losses.avg

def train_refiner(cfg, model, refiner, train_data_loader, refiner_optimizer, loss_function_refiner, epoch_idx):
    """Train the refiner for one epoch, computing both model and refiner losses."""
    model.eval()
    refiner.train()

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze refiner parameters
    for param in refiner.parameters():
        param.requires_grad = True

    # Initialize meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model_losses = AverageMeter()     # Added to track model loss
    refiner_losses = AverageMeter()

    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    batch_end_time = time.time()
    n_batches = len(train_data_loader)

    for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
        rendering_images = rendering_images[:, :n_views_rendering, ...]
        data_time.update(time.time() - batch_end_time)
        
        rendering_images = utils.helpers.var_or_cuda(rendering_images)
        ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

        # Forward pass through the model (without gradients)
        with torch.no_grad():
            output_3D, _ = model(rendering_images)
            # Compute model loss
            model_loss = loss_function_refiner(output_3D, ground_truth_volumes)
            model_losses.update(model_loss.item())

        # Forward pass through the refiner
        refined_output_3D = refiner(output_3D.detach())

        # Compute refiner loss
        refiner_loss = loss_function_refiner(refined_output_3D, ground_truth_volumes)

        # Backward pass and optimization
        refiner_optimizer.zero_grad()
        refiner_loss.backward()
        refiner_optimizer.step()

        # Update refiner loss meter
        refiner_losses.update(refiner_loss.item())

        batch_time.update(time.time() - batch_end_time)
        batch_end_time = time.time()

        # Logging
        if batch_idx == 0 or (batch_idx + 1) % cfg.TRAIN.SHOW_TRAIN_STATE == 0:
            utils.logging.info(
                f"[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}]"
                f"[Batch {batch_idx + 1}/{n_batches}] "
                f"BatchTime = {batch_time.val:.3f} (s) "
                f"DataTime = {data_time.val:.3f} (s) "
                f"ModelLoss = {model_losses.val:.4f} "
                f"RefinerLoss = {refiner_losses.val:.4f}"
            )

    # Optionally, you can return both average losses
    return model_losses.avg, refiner_losses.avg


best_model_loss = float('inf')
best_refiner_loss = float('inf')

def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    # Load data
    train_data_loader = pipeline.load_train_data(cfg)

    # Load models
    model = voxelNet(cfg)
    refiner = Refiner(cfg)

    # Initialize training parameters
    init_epoch, model, cfg = pipeline.setup_network(cfg, model)
    init_epoch, refiner, cfg = pipeline.setup_refiner(cfg, refiner)

    test_net(cfg, model, refiner)

    # Set up optimizers and loss functions
    loss_function_model = get_loss_function(cfg.TRAIN.LOSS)
    loss_function_refiner = get_loss_function(cfg.TRAIN.LOSS)

    if cfg.USE_REFINER:
        # Only train the refiner
        refiner_optimizer = torch.optim.Adam(refiner.parameters(), lr=0.0001)
        model_optimizer = None
    else:
        # Only train the model
        model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        refiner_optimizer = None

    # Training loop
    for epoch_idx in range(0, cfg.TRAIN.NUM_EPOCHS):
        epoch_start_time = time.time()

        if cfg.USE_REFINER:
            # Train refiner and get both average losses
            avg_model_loss, avg_refiner_loss = train_refiner(
                cfg, model, refiner, train_data_loader, refiner_optimizer,
                loss_function_refiner, epoch_idx
            )
        else:
            # Train model and get both average losses
            avg_model_loss, avg_refiner_loss = train_model(
                cfg, model, refiner, train_data_loader, model_optimizer,
                loss_function_model, epoch_idx
            )

        utils.logging.info(
            f"[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}] "
            f"EpochTime = {time.time() - epoch_start_time:.3f} (s) "
            f"ModelLoss = {avg_model_loss:.4f} "
            f"RefinerLoss = {avg_refiner_loss:.4f}"
        )

        # Save weights if necessary
        if (epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or
            avg_model_loss < best_model_loss or avg_refiner_loss < best_refiner_loss):
            iou = test_net(cfg, epoch_idx + 1, None, None, model, refiner)
            pipeline.save_checkpoint(cfg, epoch_idx, iou, model, refiner, avg_model_loss, avg_refiner_loss)

            if cfg.TEST.RUN_FSCORE:
                utils.logging.info("Running test_net_fscore...")
                test_net_fscore(cfg, epoch_idx + 1, val_data_loader, val_file_num, model, refiner, avg_model_loss, avg_refiner_loss)
            else:
                utils.logging.info("Skipping test_net_fscore.")