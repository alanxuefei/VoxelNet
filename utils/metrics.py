import numpy as np
import open3d
import torch
from skimage import measure  # Import marching cubes from skimage

def voxel_grid_to_mesh(vox_grid: np.array) -> open3d.geometry.TriangleMesh:
    """
    Converts a voxel grid represented as a numpy array into a mesh.
    """
    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    
    padded_grid = np.pad(vox_grid, ((1, 1), (1, 1), (1, 1)), 'constant')
    
    # Check the range of values in the padded_grid
    min_val, max_val = padded_grid.min(), padded_grid.max()
    if min_val == max_val:
        raise ValueError("Voxel grid does not contain sufficient variation for surface extraction.")
    
    # Set the level to the midpoint of the voxel grid's range
    level = 0.5 * (min_val + max_val)
    
    verts, faces, _, _ = measure.marching_cubes(padded_grid, level=level)  # Adjust level if necessary
    verts = verts / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(verts)
    out_mesh.triangles = open3d.utility.Vector3iVector(faces)
    return out_mesh

def calculate_iou(pred_vol: torch.Tensor, gt_vol: torch.Tensor) -> torch.Tensor:
    """
        Cacluate Intersection-over-Union Score
    :param pred_vol: Predicted volume, shape: [B, N_VOX (32), N_VOX (32), N_VOX (32)]
    :param gt_vol: Ground-truth volume, shape: [B, N_VOX (32), N_VOX (32), N_VOX (32)]
    :return: IoU Score, scalar
    """
    dims = [1, 2, 3]
    intersection = (pred_vol * gt_vol).float().sum(dims)
    union = ((pred_vol + gt_vol) >= 1).float().sum(dims)
    return (intersection / union).mean()


def calculate_dice(pred_vol: torch.Tensor, gt_vol: torch.Tensor) -> torch.Tensor:
    """
        Cacluate Dice Score
    :param pred_vol: Predicted volume, shape: [B, N_VOX (32), N_VOX (32), N_VOX (32)]
    :param gt_vol: Ground-truth volume, shape: [B, N_VOX (32), N_VOX (32), N_VOX (32)]
    :return: Dice Score, scalar
    """
    dims = [1, 2, 3]
    intersection = torch.sum(pred_vol * gt_vol, dim=dims).float()
    union = torch.sum(pred_vol, dim=dims) + torch.sum(gt_vol, dim=dims)
    return torch.mean((2. * intersection) / union)


def calculate_occupation_ratio(pred_vol: torch.Tensor, gt_vol: torch.Tensor) -> torch.Tensor:
    """
        Cacluate the ratio of the occupied voxels between predicted and ground-truth volumes
        When combined with other metrics, like IoU, helps to understand
            if the network predicts more or less occupied voxels than the ground-truth
    :param pred_vol: Predicted volume, shape: [B, N_VOX (32), N_VOX (32), N_VOX (32)]
    :param gt_vol: Ground-truth volume, shape: [B, N_VOX (32), N_VOX (32), N_VOX (32)]
    :return: Occupation Ratio, scalar
    """
    dims = [1, 2, 3]
    num_occupied_pred = pred_vol.float().sum(dims)
    num_occupied_gt = gt_vol.float().sum(dims)
    return (num_occupied_pred / num_occupied_gt).mean()


# def voxel_grid_to_mesh(vox_grid: np.array) -> open3d.geometry.TriangleMesh:
#     """
#         taken from: https://github.com/lmb-freiburg/what3d

#         Converts a voxel grid represented as a numpy array into a mesh.
#     """
#     sp = vox_grid.shape
#     if len(sp) != 3 or sp[0] != sp[1] or \
#             sp[1] != sp[2] or sp[0] == 0:
#         raise ValueError("Only non-empty cubic 3D grids are supported.")
#     padded_grid = np.pad(vox_grid, ((1, 1), (1, 1), (1, 1)), 'constant')
#     m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
#     m_vert = m_vert / (padded_grid.shape[0] - 1)
#     out_mesh = open3d.geometry.TriangleMesh()
#     out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
#     out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
#     return out_mesh


def calculate_fscore(list_pr: np.array, list_gt: np.array, th: float = 0.01) -> float:
    """
        based on: https://github.com/lmb-freiburg/what3d

        Calculates the F-score between two point clouds with the corresponding threshold value.
    """
    num_sampled_pts = 8192
    assert list_pr.shape == list_gt.shape
    b_size = list_gt.shape[0]

    list_gt, list_pr = list_gt.detach().cpu().numpy(), list_pr.detach().cpu().numpy()

    result = []

    for i in range(b_size):
        gt, pr = list_gt[i], list_pr[i]

        if (gt.sum() == 0 and pr.sum() != 0) or (gt.sum() != 0 and pr.sum() == 0):
            result.append(0)
            continue

        gt = voxel_grid_to_mesh(gt).sample_points_uniformly(num_sampled_pts)
        pr = voxel_grid_to_mesh(pr).sample_points_uniformly(num_sampled_pts)

        d1 = gt.compute_point_cloud_distance(pr)
        d2 = pr.compute_point_cloud_distance(gt)

        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
        result.append(fscore)

    return np.array(result)
    