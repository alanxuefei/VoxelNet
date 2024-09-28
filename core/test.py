import numpy as np
import torch
import utils.metrics
import matplotlib.pyplot as plt
from utils import logging
from datetime import datetime as dt
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import os
import cv2

import core.pipeline_test as pipeline

from models.voxel_net.voxel_net import voxelNet
from models.voxel_net.refiner import Refiner
from losses.losses import DiceLoss, CEDiceLoss, FocalLoss
from utils.average_meter import AverageMeter

def test_net(cfg,
             epoch_idx,
             test_data_loader,
             test_file_num,
             model,
             refiner):
    torch.backends.cudnn.benchmark = True
    # Load data
    taxonomies, test_data_loader, test_file_num = pipeline.load_data(cfg, test_data_loader, test_file_num)

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = []
    taxonomies_list = []

    model.eval()
    refiner.eval()

    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume).to(torch.cuda.current_device())

            # Test the model
            generated_volume, _ = model(rendering_images) 
            generated_volume = generated_volume.squeeze(dim=1)
            # Pass through the refiner model
            generated_volume = refiner(generated_volume).clamp_max(1)

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).unsqueeze(dim=0))
            test_iou_value = torch.cat(sample_iou).mean().item()  # Taking the mean IoU over thresholds

            # Update test IoU results
            test_iou.append(torch.cat(sample_iou).unsqueeze(dim=0))
            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

            # Print sample loss and IoU
            if (sample_idx + 1) % 50 == 0:
                for_tqdm.update(50)
                for_tqdm.set_description('Test[%d/%d] Taxonomy = %s' %
                                            (sample_idx + 1, n_samples, taxonomy_id))

            logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s IoU = %s' %
                            (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                            ['%.4f' % si for si in sample_iou]))

    test_iou = torch.cat(test_iou, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    test_iou = pipeline.combine_test_iou(test_iou, taxonomies_list, list(taxonomies.keys()), test_file_num)

    # Output testing results
    mean_iou = pipeline.output(cfg, test_iou, taxonomies)

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)

    print('The IoU score of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou))

    return max_iou

def test_net_fscore(cfg,
                    epoch_idx=-1,
                    test_data_loader=None,
                    test_file_num=None,
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
            generated_volume, attn_weights = model(rendering_images)
            generated_volume = generated_volume.squeeze(dim=1)
            save_attention_map_overlay(attn_weights, rendering_images, sample_name)
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

        print('The IoU score before refiner of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou_before))
        print('The IoU score after refiner of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou_after))
        print('The mean F-score before refiner is %.4f\n' % mean_fscore_before)
        print('The mean F-score after refiner is %.4f\n' % mean_fscore_after)

        return max_iou_after


def calculate_iou(predicted_volume, ground_truth_volume, thresholds):
    """
    Calculate IoU between the predicted volume and ground truth across multiple thresholds.
    
    Args:
        predicted_volume: The predicted 3D volume output.
        ground_truth_volume: The ground truth 3D volume.
        thresholds: A list of voxel thresholds to compute IoU at multiple levels.
    
    Returns:
        sample_iou: A list of IoU values for each threshold.
    """
    sample_iou = []
    for th in thresholds:
        _volume = torch.ge(predicted_volume, th).float()  # Threshold the predicted volume
        intersection = torch.sum(_volume.mul(ground_truth_volume)).float()  # Intersection
        union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()  # Union
        sample_iou.append((intersection / union).unsqueeze(dim=0))  # IoU
    return torch.cat(sample_iou).mean().item()  # Mean IoU over all thresholds


import os
import logging
import numpy as np
import torch
import cv2

def attention_rollout(attn_weights_list):
    """
    This function performs the attention rollout by recursively multiplying the attention weights across all layers.
    
    Parameters:
    - attn_weights_list: A list of attention weights across different layers.
    
    Returns:
    - A single attention map that represents the contribution of each input token to the output.
    """
    # Start with an identity matrix of size (n_tokens, n_tokens)
    rollout = torch.eye(attn_weights_list[0].size(-1)).to(attn_weights_list[0].device)
    
    for i, attn_weights in enumerate(attn_weights_list):
        # Take the mean of attention across all heads
        attn_weights_mean = attn_weights.mean(dim=1)  # Shape: (batch_size, num_views, num_views)
        
        # Add a skip connection to account for the identity mapping in transformers
        attn_weights_mean = attn_weights_mean + torch.eye(attn_weights_mean.size(-1)).to(attn_weights_mean.device)
        
        # Normalize the weights to make them a probability distribution
        attn_weights_mean = attn_weights_mean / attn_weights_mean.sum(dim=-1, keepdim=True)

        # Debug: Print the min and max values of the attention weights for each layer
        logging.info(f"Layer {i}: min attention weight = {attn_weights_mean.min().item()}, max attention weight = {attn_weights_mean.max().item()}")
        
        # Multiply the current rollout with the new attention weights
        rollout = torch.matmul(attn_weights_mean, rollout)
    
    return rollout

import os
import logging
import torch
import numpy as np
import cv2

def save_attention_map_overlay(attn_weights, rendering_images, sample_name, save_dir="attn_maps"):
    """
    This function saves the attention maps overlaid on the original images for each view using a grayscale color scale.

    Parameters:
    - attn_weights: The attention weights generated by the model.
    - rendering_images: The original input images to the model.
    - sample_name: The name of the sample being processed.
    - save_dir: The directory to save the attention maps.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Created directory {save_dir} for saving attention maps.")

    # Log input shapes
    logging.info(f"Rendering images shape: {rendering_images.shape}")
    logging.info(f"Attention weights shape before processing: {attn_weights.shape}")

    batch_size, num_views, channels, height, width = rendering_images.shape

    if attn_weights.dim() == 3:
        logging.info("Processing 3D attention weights.")
        for i in range(batch_size):
            decoupled_attn_weights = torch.zeros((num_views, height, width)).to(attn_weights.device)
            for v in range(num_views):
                logging.info(f"Processing sample {sample_name}, view {v}")

                attn_map = attn_weights[i, v]  # Shape: (num_views)

                for view_idx in range(num_views):
                    decoupled_attn_weights[view_idx] += attn_map[view_idx]

            for view_idx in range(num_views):
                attn_map = decoupled_attn_weights[view_idx]
                attn_map_min = attn_map.min().item()
                attn_map_max = attn_map.max().item()
                logging.info(f"View {view_idx}: Attention map min = {attn_map_min}, max = {attn_map_max}")

                if attn_map_max > attn_map_min:
                    attn_map = (attn_map - attn_map_min) / (attn_map_max - attn_map_min)

                # Invert the attention map for dark-to-light scaling
                attn_map = 1 - attn_map  # Now, higher values mean less attention (darker)

                # Move tensor to CPU and convert to NumPy array
                attn_map = attn_map.cpu().numpy()

                # Scale the attention values to increase contrast
                attn_map = np.clip(attn_map * 1.5, 0, 1)

                # Convert to grayscale intensity (0 = black, 255 = white)
                attn_map_resized = (attn_map * 255).astype(np.uint8)

                # Convert to BGR for visualizing the overlay
                attn_map_resized = cv2.cvtColor(attn_map_resized, cv2.COLOR_GRAY2BGR)

                img = rendering_images[i, view_idx].cpu().numpy().transpose(1, 2, 0)  # (height, width, channels)
                img = (img * 255).astype(np.uint8)

                # Overlay the attention map on the image with increased opacity
                overlay = cv2.addWeighted(img, 0.6, attn_map_resized, 0.4, 0)

                save_path = os.path.join(save_dir, f"{sample_name}_view{view_idx}_attn.png")
                cv2.imwrite(save_path, overlay)
                logging.info(f"Saved attention map overlay for sample {sample_name}, view {view_idx} to {save_path}")

    else:
        logging.error(f"Unexpected dimensions for attn_weights: {attn_weights.dim()}D tensor. Expected 3D tensor.")

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
