import numpy as np
import torch
import utils.metrics
#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

from utils import logging
from datetime import datetime as dt
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.data_loaders
import utils.data_transforms
import utils.helpers

import core.pipeline_test as pipeline

from models.voxel_net.voxel_net import voxelNet
from models.voxel_net.refiner import Refiner
from losses.losses import DiceLoss, CEDiceLoss, FocalLoss
from utils.average_meter import AverageMeter
def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_file_num=None,
             test_writer=None,
             model=None,
             refiner=None):  # Added refiner as a parameter
    torch.backends.cudnn.benchmark = True

    # Load data
    taxonomies, test_data_loader, test_file_num = pipeline.load_data(cfg, test_data_loader, test_file_num)

    # Set up model
    if model is None:
        model = voxelNet(
            img_size=cfg.CONST.IMG_H,
            in_chans=3,
            embed_dim=cfg.NETWORK.EMBED_DIM,
            combined_dim=cfg.NETWORK.COMBINED_DIM,
            output_shape=cfg.NETWORK.OUTPUT_SHAPE,
            num_views=cfg.CONST.N_VIEWS_RENDERING
        )
        model, epoch_idx = pipeline.setup_network(cfg, model)
    
    # Set up loss functions
    if cfg.TRAIN.LOSS == 1:
        loss_function = torch.nn.BCELoss()
    elif cfg.TRAIN.LOSS == 2:
        loss_function = DiceLoss()
    elif cfg.TRAIN.LOSS == 3:
        loss_function = CEDiceLoss()
    elif cfg.TRAIN.LOSS == 4:
        loss_function = FocalLoss()


    iou_results = {}
    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = []
    taxonomies_list = []
    losses = AverageMeter()

    model.eval()
    refiner.eval()  # Set refiner to evaluation mode

    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume).to(torch.cuda.current_device())

            # Test the model
            generated_volume = model(rendering_images).squeeze(dim=1)

            # Pass through the refiner model
            refined_volume = refiner(generated_volume).clamp_max(1)

            # Loss
            loss = loss_function(refined_volume, ground_truth_volume)

            # Append loss and accuracy to average metrics
            loss = utils.helpers.reduce_value(loss)
            losses.update(loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(refined_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).unsqueeze(dim=0))
            test_iou_value = torch.cat(sample_iou).mean().item()  # Taking the mean IoU over thresholds

            # Update test IoU results
            test_iou.append(torch.cat(sample_iou).unsqueeze(dim=0))
            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

            # save_iou_results(taxonomy_id, sample_name, test_iou_value, generated_volume, refined_volume, ground_truth_volume, "/workspace/prediction", taxonomies)

            if True:
                # Print sample loss and IoU
                if (sample_idx + 1) % 50 == 0:
                    for_tqdm.update(50)
                    for_tqdm.set_description('Test[%d/%d] Taxonomy = %s Loss = %.4f' %
                                             (sample_idx + 1, n_samples, taxonomy_id, losses.avg))

                logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s Loss = %.4f IoU = %s' %
                              (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                               loss.item(), ['%.4f' % si for si in sample_iou]))

    test_iou = torch.cat(test_iou, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    test_iou = pipeline.combine_test_iou(test_iou, taxonomies_list, list(taxonomies.keys()), test_file_num)

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    if True:
        # Output testing results
        mean_iou = pipeline.output(cfg, test_iou, taxonomies)

        # Add testing results to TensorBoard
        max_iou = np.max(mean_iou)
        if test_writer is not None:
            test_writer.add_scalar('EpochLoss', losses.avg, epoch_idx)
            test_writer.add_scalar('IoU', max_iou, epoch_idx)

        print('The IoU score of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou))

        return max_iou

def test_net_fscore(cfg,
                    epoch_idx=-1,
                    test_data_loader=None,
                    test_file_num=None,
                    test_writer=None,
                    model=None,
                    refiner=None):
    torch.backends.cudnn.benchmark = True

    # Load data
    taxonomies, test_data_loader, test_file_num = pipeline.load_data(cfg, test_data_loader, test_file_num)

    # Set up model
    if model is None:
        model = voxelNet(
            img_size=cfg.CONST.IMG_H,
            in_chans=3,
            embed_dim=cfg.NETWORK.EMBED_DIM,
            combined_dim=cfg.NETWORK.COMBINED_DIM,
            output_shape=cfg.NETWORK.OUTPUT_SHAPE,
            num_views=cfg.CONST.N_VIEWS_RENDERING
        )
        model, epoch_idx = pipeline.setup_network(cfg, model)
    
    # Set up loss functions
    if cfg.TRAIN.LOSS == 1:
        loss_function = torch.nn.BCELoss()
    elif cfg.TRAIN.LOSS == 2:
        loss_function = DiceLoss()
    elif cfg.TRAIN.LOSS == 3:
        loss_function = CEDiceLoss()
    elif cfg.TRAIN.LOSS == 4:
        loss_function = FocalLoss()

    iou_results = {}
    n_samples = len(test_data_loader)
    test_iou_before = []
    test_iou_after = []
    test_fscore_before = []  # Add F-score tracking for before refiner
    test_fscore_after = []   # Add F-score tracking for after refiner
    taxonomies_list = []
    losses = AverageMeter()

    model.eval()
    refiner.eval()  # Set refiner to evaluation mode

    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume).to(torch.cuda.current_device())

            # Test the model
            generated_volume = model(rendering_images).squeeze(dim=1)

            # Calculate IoU before refiner
            sample_iou_before = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou_before.append((intersection / union).unsqueeze(dim=0))
            iou_before_value = torch.cat(sample_iou_before).mean().item()  # Taking the mean IoU over thresholds
            test_iou_before.append(torch.cat(sample_iou_before).unsqueeze(dim=0))

            # Calculate F-score before refiner
            fscore_before_value = utils.metrics.calculate_fscore(generated_volume, ground_truth_volume).mean().item()
            test_fscore_before.append(fscore_before_value)

            # Pass through the refiner model
            refined_volume = refiner(generated_volume).clamp_max(1)

            # Calculate IoU after refiner
            sample_iou_after = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(refined_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou_after.append((intersection / union).unsqueeze(dim=0))
            iou_after_value = torch.cat(sample_iou_after).mean().item()  # Taking the mean IoU over thresholds
            test_iou_after.append(torch.cat(sample_iou_after).unsqueeze(dim=0))

            # Calculate F-score after refiner
            fscore_after_value = utils.metrics.calculate_fscore(refined_volume, ground_truth_volume).mean().item()
            test_fscore_after.append(fscore_after_value)

            # Loss
            loss = loss_function(refined_volume, ground_truth_volume)

            # Append loss and accuracy to average metrics
            loss = utils.helpers.reduce_value(loss)
            losses.update(loss.item())

            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

            if True:
                # Print sample loss, IoU, and F-score
                if (sample_idx + 1) % 50 == 0:
                    for_tqdm.update(50)
                    for_tqdm.set_description('Test[%d/%d] Taxonomy = %s Loss = %.4f' %
                                             (sample_idx + 1, n_samples, taxonomy_id, losses.avg))

                logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s Loss = %.4f IoU before = %.4f IoU after = %.4f F-score before = %.4f F-score after = %.4f' %
                              (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                               loss.item(), iou_before_value, iou_after_value, fscore_before_value, fscore_after_value))

    test_iou_before = torch.cat(test_iou_before, dim=0)
    test_iou_after = torch.cat(test_iou_after, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    test_iou_before = pipeline.combine_test_iou(test_iou_before, taxonomies_list, list(taxonomies.keys()), test_file_num)
    test_iou_after = pipeline.combine_test_iou(test_iou_after, taxonomies_list, list(taxonomies.keys()), test_file_num)

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    if True:
        # Output testing results
        mean_iou_before = pipeline.output(cfg, test_iou_before, taxonomies)
        mean_iou_after = pipeline.output(cfg, test_iou_after, taxonomies)

        # Calculate and output mean F-score
        mean_fscore_before = np.mean(test_fscore_before)
        mean_fscore_after = np.mean(test_fscore_after)

        # Add testing results to TensorBoard
        max_iou_before = np.max(mean_iou_before)
        max_iou_after = np.max(mean_iou_after)
        if test_writer is not None:
            test_writer.add_scalar('EpochLoss', losses.avg, epoch_idx)
            test_writer.add_scalar('IoU before', max_iou_before, epoch_idx)
            test_writer.add_scalar('IoU after', max_iou_after, epoch_idx)
            test_writer.add_scalar('F-score before', mean_fscore_before, epoch_idx)
            test_writer.add_scalar('F-score after', mean_fscore_after, epoch_idx)

        print('The IoU score before refiner of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou_before))
        print('The IoU score after refiner of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou_after))
        print('The mean F-score before refiner is %.4f\n' % mean_fscore_before)
        print('The mean F-score after refiner is %.4f\n' % mean_fscore_after)

        return max_iou_after


def batch_test(cfg):
    import sys
    import os

    dir_name, _ = os.path.split(cfg.CONST.WEIGHTS)
    if True:
        log_file = os.path.join(dir_name, 'test_log_%s.txt' % dt.now().isoformat())
        f = open(log_file, 'w')
        sys.stdout = f

    view_num_list = [1, 2, 3, 4, 5, 8, 12, 16, 20]
    for view_num in view_num_list:
        cfg.CONST.N_VIEWS_RENDERING = view_num
        test_net(cfg)

    if True:
        f.close()
