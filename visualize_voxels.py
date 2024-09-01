import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.binvox_visualization import get_volume_views

def load_voxel_grid(file_path):
    """
    Load a voxel grid from a .npy file.
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"Voxel grid file not found: {file_path}")

def plot_voxel_views(taxonomy_name, iou, pred_voxel_images, refined_voxel_images, gt_voxel_images, output_file):
    """
    Plot voxel images side by side for predicted, refined, and ground truth voxel grids,
    and save the plot to a file.
    """
    num_angles = len(pred_voxel_images)
    fig, axes = plt.subplots(3, num_angles, figsize=(15, 8))
    for i, (pred_img, refined_img, gt_img) in enumerate(zip(pred_voxel_images, refined_voxel_images, gt_voxel_images)):
        axes[0, i].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{taxonomy_name} - Predicted {i+1})')
        
        axes[1, i].imshow(cv2.cvtColor(refined_img, cv2.COLOR_BGR2RGB))
        axes[1, i].axis('off')
        axes[1, i].set_title(f'{taxonomy_name} - Refined {i+1} (IoU: {iou:.4f})')
        
        axes[2, i].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        axes[2, i].axis('off')
        axes[2, i].set_title(f'{taxonomy_name} - Ground Truth {i+1}')
    
    plt.savefig(output_file)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory

def visualize_voxels(taxonomy_name, iou, predicted_voxel_file, refined_voxel_file, ground_truth_voxel_file, angles, output_file):
    """
    Visualize the predicted, refined, and ground truth voxel grids and save the images.
    """
    # Load voxel grids
    predicted_voxel = load_voxel_grid(predicted_voxel_file)
    refined_voxel = load_voxel_grid(refined_voxel_file)
    ground_truth_voxel = load_voxel_grid(ground_truth_voxel_file)

    # Get images of the voxel grids
    pred_voxel_images = get_volume_views(predicted_voxel, '/workspace/prediction/', 0, angles)
    refined_voxel_images = get_volume_views(refined_voxel, '/workspace/prediction/', 2, angles)
    gt_voxel_images = get_volume_views(ground_truth_voxel, '/workspace/prediction/', 1, angles)

    # Save images side by side with updated labels
    plot_voxel_views(taxonomy_name, iou, pred_voxel_images, refined_voxel_images, gt_voxel_images, output_file)

def main():
    # Path to the JSON file
    json_file = '/workspace/prediction/iou_results.json'
    
    # Angles for visualization
    angles = [(45, 0), (45, 45), (45, 90)]
    
    # Load IoU results from the JSON file
    with open(json_file, 'r') as f:
        iou_results = json.load(f)
    
    # Iterate through each taxonomy and object ID in the IoU results
    for taxonomy_name, objects in iou_results.items():
        print(f"Processing category: {taxonomy_name}")
        
        # Extract and sort the IoU data for each object
        sorted_objects = sorted(objects.items(), key=lambda item: item[1]["iou"])
        
        # Get indices for best, middle, and worst IoU
        best_idx = -1
        worst_idx = 0
        middle_idx = len(sorted_objects) // 2

        for idx, label in zip([best_idx, middle_idx, worst_idx], ["best", "middle", "worst"]):
            object_id, data = sorted_objects[idx]
            iou = data["iou"]

            # File paths should already be correct since the JSON file has full paths
            predicted_voxel_file = data["generated_voxel"]
            refined_voxel_file = data["refined_voxel"]
            ground_truth_voxel_file = data["ground_truth_voxel"]

            # Print IoU and file paths
            print(f"Object ID: {object_id}")
            print(f"IoU: {iou}")
            print(f"Predicted voxel file: {predicted_voxel_file}")
            print(f"Refined voxel file: {refined_voxel_file}")
            print(f"Ground truth voxel file: {ground_truth_voxel_file}")
            print("-" * 30)

            # Generate output file name including taxonomy name, object ID, and IoU value
            output_file = f"/workspace/selection/view_{taxonomy_name}_{object_id}_{label}_iou_{iou:.4f}.png"

            # Visualize and save the voxels for the current object
            visualize_voxels(taxonomy_name, iou, predicted_voxel_file, refined_voxel_file, ground_truth_voxel_file, angles, output_file)

if __name__ == '__main__':
    main()
