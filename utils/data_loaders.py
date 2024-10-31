#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox

import cv2
import json
import numpy as np
from utils import logging
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from enum import Enum, unique

import utils.binvox_rw

import pickle


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                logging.error('It seems that there is something wrong with the image file %s' % image_path)
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())
    
    def save_files_to_disk(self, files, filename='shapenet_files.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(files, f)
        logging.info(f'Files saved to {filename}')
    
    def load_files_from_disk(self, filename='shapenet_files.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                files = pickle.load(f)
            logging.info(f'Files loaded from {filename}')
            return files
        else:
            logging.info(f'No file list found at {filename}')
            return None

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None, cache_file='shapenet_files.pkl'):
        """
        Retrieves the ShapeNet dataset for the specified dataset type, storing all train, test, and val files into 'files'.
        The files are accessible via files[DatasetType.TRAIN], files[DatasetType.TEST], files[DatasetType.VAL].

        Args:
            dataset_type (DatasetType): The type of dataset to retrieve (e.g., DatasetType.TRAIN, DatasetType.TEST, DatasetType.VAL).
            n_views_rendering (int): Number of views for rendering.
            transforms (callable, optional): Optional transformations to apply to the dataset.
            cache_file (str, optional): Path to the cache file. Defaults to 'shapenet_files.pkl'.

        Returns:
            tuple: A tuple containing the ShapeNetDataset instance and the number of files.
        """
        # Load cached files from disk
        files = self.load_files_from_disk(cache_file)

        # Initialize the files dictionary if cache is empty
        if files is None:
            files = {}
            logging.info('Cache is empty. Initializing new cache.')

            # Initialize entries for each dataset type
            files[DatasetType.TRAIN] = []
            files[DatasetType.TEST] = []
            files[DatasetType.VAL] = []

            # Load data for each taxonomy and dataset type
            for taxonomy in self.dataset_taxonomy:
                taxonomy_id = taxonomy['taxonomy_id']
                taxonomy_name = taxonomy['taxonomy_name']
                logging.info('Collecting files for Taxonomy[ID=%s, Name=%s]' %
                            (taxonomy_id, taxonomy_name))

                # Collect files for each dataset type
                for dt in [DatasetType.TRAIN, DatasetType.TEST, DatasetType.VAL]:
                    # Get the sample list for the current dataset type
                    samples = taxonomy.get(dt.name.lower(), [])
                    if not samples:
                        continue  # Skip if there are no samples for this dataset type

                    logging.info('Collecting %d samples for DatasetType=%s' % (len(samples), dt.name))

                    # Get files for the current taxonomy and dataset type
                    taxonomy_files = self.get_files_of_taxonomy(taxonomy_id, samples)

                    # Extend the files list for the current dataset type
                    files[dt].extend(taxonomy_files)

                    logging.info('Collected %d files for DatasetType=%s' % (len(taxonomy_files), dt.name))

            # Save the updated files dictionary to disk
            self.save_files_to_disk(files, cache_file)
            logging.info('Files collected and saved to cache.')

        else:
            logging.info('Using cached files.')

        # Get files for the requested dataset_type
        all_files = files.get(dataset_type, [])
        logging.info('Total files collected for DatasetType=%s: %d.' % (dataset_type.name, len(all_files)))

        # Return the dataset and the number of files
        return ShapeNetDataset(dataset_type, all_files, n_views_rendering, transforms), len(all_files)


    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                logging.warn('Ignore sample %s/%s since image files not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #


class Pix3dDataset(torch.utils.data.dataset.Dataset):
    """Pix3D class used for PyTorch DataLoader"""
    
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, bounding_box = self.get_datum(idx)
        
        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)
        
        return taxonomy_name, sample_name, rendering_images, volume
    
    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        bounding_box = self.file_list[idx]['bounding_box']
        volume_path = self.file_list[idx]['volume']
        
        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        
        if len(rendering_image.shape) < 3:
            rendering_image = np.stack((rendering_image,) * 3, -1)
        
        if rendering_image.shape[-1] == 4:
            rendering_image = cv2.cvtColor(rendering_image, cv2.COLOR_BGRA2BGR)
        
        # Get data of volume
        volume = scipy.io.loadmat(volume_path)
        volume = volume['voxel'].astype(np.float32)
        volume = volume.transpose(1, 2, 0)
        
        return taxonomy_name, sample_name, np.asarray([rendering_image]), volume, bounding_box


# //////////////////////////////// = End of Pix3dDataset Class Definition = ///////////////////////////////// #


class Pix3dDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.annotations = dict()
        self.volume_path_template = cfg.DATASETS.PIX3D.VOXEL_PATH
        self.rendering_image_path_template = cfg.DATASETS.PIX3D.RENDERING_PATH
        
        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.PIX3D.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())
        
        # Load all annotations of the dataset
        _annotations = None
        with open(cfg.DATASETS.PIX3D.ANNOTATION_PATH, encoding='utf-8') as file:
            _annotations = json.loads(file.read())
        
        for anno in _annotations:
            filename, _ = os.path.splitext(anno['img'])
            anno_key = filename[4:]
            self.annotations[anno_key] = anno

    def load_files_from_disk(self):
        if os.path.exists(self.file_list_path):
            with open(self.file_list_path, 'rb') as f:
                files = pickle.load(f)
            logging.info('Files have been loaded from disk.')
            return files
        else:
            logging.info('No file list found on disk.')
            return None
        
    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = self.load_files_from_disk()

        if files is None:
            files = []
            # Load data for each category
            for taxonomy in self.dataset_taxonomy:
                taxonomy_name = taxonomy['taxonomy_name']
                logging.info('Collecting files of Taxonomy[Name=%s]' % (taxonomy_name))

                samples = []
                if dataset_type == DatasetType.TRAIN:
                    samples = taxonomy['train']
                elif dataset_type == DatasetType.TEST:
                    samples = taxonomy['test']
                elif dataset_type == DatasetType.VAL:
                    samples = taxonomy['test']

                files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

            logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
            self.save_files_to_disk(files)

        return Pix3dDataset(files, transforms)
    
    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []
        
        for sample_idx, sample_name in enumerate(samples):
            # Get image annotations
            anno_key = '%s/%s' % (taxonomy_name, sample_name)
            annotations = self.annotations[anno_key]
            
            # Get file list of rendering images
            _, img_file_suffix = os.path.splitext(annotations['img'])
            rendering_image_file_path = self.rendering_image_path_template % \
                                        (taxonomy_name, sample_name, img_file_suffix[1:])
            
            # Get the bounding box of the image
            img_width, img_height = annotations['img_size']
            bbox = [
                annotations['bbox'][0] / img_width,
                annotations['bbox'][1] / img_height,
                annotations['bbox'][2] / img_width,
                annotations['bbox'][3] / img_height
            ]  # yapf: disable
            
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (annotations['voxel'][:-4] + '_32.mat')
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_name, sample_name))
                continue
            
            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })
        
        return files_of_taxonomy


# /////////////////////////////// = End of Pix3dDataLoader Class Definition = /////////////////////////////// #


DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'Pix3D': Pix3dDataLoader,
}  # yapf: disable
