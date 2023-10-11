from tools.dataset import *
from tools.lastools import *

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script takes a single point cloud and creates batches with a specific number of points
"""

# Define Path where the point cloud is stored. Possible format -> LAS or LAZ
path = r'../Daten/Datensatz_H3D/DIM_2022/7 - DBScan/edited/554000_5800000.laz'

# set batchsize --> number of points for a single batch
num_points = 100000

# create a Dataset object --> defined in tools/dataset
dataset = Dataset(path)

# calculate the spacing between the batches with the formula of Winiwartet et al. (2018)
spacing = np.sqrt(num_points / dataset.pointdensity / np.pi) * np.sqrt(2) / 2 * 0.95

# Define witch attributes we want to keep for the batches
atts = ['X', 'Y', 'Z', 'intensity', 'return_number', 'red', 'green', 'blue',
        'hue', 'saturation', 'value', 'classification', 'point_index']

# initialize a kNN Batch Dataset Object --> defined in tools/dataset
kNN_Dataset = kNNBatchDataset(file=path, k=num_points, spacing=spacing, attributes=atts)

# use function to create the batches and save them to the defined path
kNN_Dataset.save_batches(r'../Daten/Datensatz_H3D/DIM_2022/8 - PointNet++/batches')
