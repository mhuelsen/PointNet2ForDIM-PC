import numpy
import numpy as np
import pandas as pd
import torch

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script contains the functions to do the dataset augmentation
"""


def z_jittering(xyz: torch.tensor, prob: float = 0.5, stddev: float = 0.05):
    # implementation for torch tenors
    if 'torch' in str(type(xyz)):
        if torch.rand(1).item() < prob:
            xyz[:, 2] += torch.normal(mean=0, std=torch.full((xyz.shape[0],), stddev))
            return xyz
        else:
            return xyz
    # implementation for numpy arrays
    elif 'numpy' in str(type(xyz)):
        if np.random.uniform() < prob:
            xyz[:, 2] += np.random.normal(0, stddev, size=xyz.shape[0])
            return xyz
        else:
            return xyz
    else:
        raise Exception(f'Jittering not implemented for {type(xyz)}')


def random_colordrop(color: torch.tensor, prob: float = 0.2):
    # drop some of the color values for a percentage
    npoints = color.shape[0]

    # generate a random mask for each row
    masks = np.random.rand(npoints, 1) > prob

    # use the masks to zero out rows
    color *= masks

    return color


def hsv_jittering(hsv: torch.tensor, prob: float = 0.5,
                  hue_std: float = 0.02, sat_std: float = 0.1, value_std: float = 0.05):
    # add random noise to the HSV values

    # implementation for torch tensor
    if 'torch' in str(type(hsv)):
        if torch.rand(1).item() < prob:
            npoints = hsv.shape[0]
            hsv[:, 0] += torch.normal(mean=0, std=torch.tensor(hue_std).expand(npoints))
            hsv[:, 1] += torch.normal(mean=0, std=torch.tensor(sat_std).expand(npoints))
            hsv[:, 2] += torch.normal(mean=0, std=torch.tensor(value_std).expand(npoints))
            hsv[hsv > 1] = 1
            hsv[hsv < 0] = 0
            return hsv
        else:
            return hsv

    # implementation for numpy array
    elif 'numpy' in str(type(hsv)):
        if np.random.uniform() < prob:
            npoints = hsv.shape[0]
            hsv[:, 0] += np.random.normal(0, hue_std, size=npoints)
            hsv[:, 1] += np.random.normal(0, sat_std, size=npoints)
            hsv[:, 2] += np.random.normal(0, value_std, size=npoints)
            hsv[hsv > 1] = 1
            hsv[hsv < 0] = 0
            return hsv
        else:
            return hsv
    else:
        raise Exception(f'Jittering not implemented for {type(hsv)}')


def hsv_shift(hsv: numpy.ndarray, hue_std: float = 0.02, sat_std: float = 0.1, value_std: float = 0.05, prob=0.5):
    # shift the HSV values by a random value with follows the normal distribution
    if np.random.uniform() < prob:
        shift = np.hstack((np.random.normal(0, hue_std, size=1),
                           np.random.normal(0, sat_std, size=1),
                           np.random.normal(0, value_std, size=1)))
        hsv += shift
        hsv[hsv > 1] = 1
        hsv[hsv < 0] = 0

    return hsv


def rotate_random_around_z(xyz, angle_range=(0, 2*np.pi)):
    # rotation the point cloud by a random angle around the z-axis
    if 'torch' in str(type(xyz)):
        # get a random angle inside the range
        angle_rad = torch.FloatTensor(1).uniform_(angle_range[0], angle_range[1])
        # get sinus and cosinus
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)

        # define rotation matrix
        rot_mat = torch.tensor([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]])

        # rotate coordinates
        xyz_rot = torch.mm(xyz, rot_mat.t())

        return xyz_rot

    elif 'numpy' in str(type(xyz)):
        # reduce coords to mean
        mean = xyz.mean(axis=0)
        xyz -= mean
        # get a random angle inside the range
        angle_rad = np.random.uniform(angle_range[0], angle_range[1])
        # get sinus and cosinus
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # define rotation matrix
        rot_mat = np.array([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]])

        # rotate coordinates
        xyz_rot = np.dot(xyz, rot_mat.T)

        return xyz_rot + mean
