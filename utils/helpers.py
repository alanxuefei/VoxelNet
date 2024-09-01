# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def reduce_value(value):
    return value


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
       type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")
 
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img

def save_voxel_volume(volume, output_dir, filename):
    """
    Save a voxel volume to a .npy file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    np.save(file_path, volume.cpu().numpy())
    return file_path

def save_iou_fscore_results(taxonomy_id, sample_name, iou_before, iou_after, fscore_before, fscore_after, generated_volume, refined_volume, ground_truth_volume, output_dir, taxonomies):
    """
    Save IoU and F-score results along with generated, refined, and ground truth volumes.
    Also, manage the results file to accumulate results across calls.
    """
    taxonomy_name = taxonomies[taxonomy_id]['taxonomy_name']
        
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load existing results from the JSON file if it exists
    results_path = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as json_file:
            results = json.load(json_file)
        
    else:
        results = {}
        
    # Save the voxel volumes with taxonomy_name in the filename
    generated_path = save_voxel_volume(generated_volume, output_dir, f"{taxonomy_name}_{sample_name}_generated.npy")
    refined_path = save_voxel_volume(refined_volume, output_dir, f"{taxonomy_name}_{sample_name}_refined.npy")
    ground_truth_path = save_voxel_volume(ground_truth_volume, output_dir, f"{taxonomy_name}_{sample_name}_ground_truth.npy")

    
    # Store the IoU and F-score results along with paths to the volumes
    if taxonomy_name not in results:
        results[taxonomy_name] = {}

    results[taxonomy_name][sample_name] = {
        "iou_before": iou_before,
        "iou_after": iou_after,
        "fscore_before": fscore_before,
        "fscore_after": fscore_after,
        "generated_voxel": generated_path,
        "refined_voxel": refined_path,
        "ground_truth_voxel": ground_truth_path
    }

    # Save the updated results back to the JSON file
    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
