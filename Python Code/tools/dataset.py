import laspy
from scipy.spatial import KDTree
import colorsys
from torch.utils import data
import os
from tools.dataaugmentation import *
from tools.lastools import *
"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script defines different tools to define a point cloud Dataset

Some parts of this script were published by Winiwarter et al. (2018): 
        https://github.com/lwiniwar/alsNet/blob/master/alsNet/dataset.py 
"""


class Dataset:
    # This class represents a single point cloud
    def __init__(self, file, load=True):
        self.file = file    # file = path to pointcloud file (*.las or *.laz)
        self.df_pc = None   # variable for the DataFrame of the point cloud
        self.name = file.split('/')[-1]     # name of the point cloud
        self.xmax = self.ymax = self.xmin = self.ymin = None    # maxima and minima of the point cloud
        if load:
            # load data from file
            self.load_data()

    def load_data(self):
        # load data from path using laspy
        with laspy.open(self.file) as f:
            las = f.read()

        # read coordinates from las
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)

        # define maximum and minimal values
        self.xmin = min(x)
        self.ymin = min(y)
        self.xmax = max(x)
        self.ymax = max(y)

        # save coordinates to DataFrame
        self.df_pc = pd.DataFrame({'X': x, 'Y': y, 'Z': z}, index=np.arange(len(x)))

        # save every dimension to DataFrame
        for i in range(3, len(las.point_format.dimensions)):
            dim = las.point_format.dimensions[i].name     # dims[i]
            self.df_pc[dim] = np.array(las[dim])
        # deselect synthetic points
        self.df_pc = self.df_pc.loc[self.df_pc['synthetic'] == 0]
        # convert colorspace if there are None values
        if 'hue' not in self.df_pc.columns or 'saturation' not in self.df_pc.columns or 'value' not in self.df_pc.columns:
            self.color_RGB2HSV()

    def select_features(self, features: list = 'default'):
        # Function to select specific features from point cloud DataFrame
        if features == 'default':
            features = ['X', 'Y', 'Z', 'intensity', 'return_number', 'red', 'green', 'blue',
                        'hue', 'saturation', 'value']
        self.df_pc = self.df_pc.loc[:, features]
        return self

    def color_RGB2HSV(self):
        # Function calculate Hue, Saturation and Value and add it to point cloud DataFrame
        rgb = self.df_pc.loc[:, 'red':'blue'].to_numpy()
        rgb = rgb / 255 / 255
        hsv = np.apply_along_axis(lambda x: colorsys.rgb_to_hsv(*x), -1, rgb)
        self.df_pc[['hue', 'saturation', 'value']] = hsv

    def save_labels_to_cloud(self, labels):
        # Function to add the results of a segmentation to the point cloud DataFrame
        self.df_pc['pred_classification'] = labels
        return self.df_pc

    @property
    def labels(self):
        # get labels as numpy array
        if self.df_pc is None:
            self.load_data()
        labels = self.df_pc.loc[:, 'classification'].to_numpy()
        return labels

    @property
    def xyz(self):
        # get coordinates (XYZ) as numpy array
        if self.df_pc is None:
            self.load_data()
        xyz = self.df_pc.loc[:, 'X':'Z'].to_numpy()
        return xyz

    @property
    def features(self):
        # get features a numpy array
        if self.df_pc is None:
            self.load_data()

        feat = self.df_pc.drop(['X', 'Y', 'Z', 'classification', 'point_index', 'red', 'green', 'blue'],
                               axis=1, errors='ignore').to_numpy()
        return feat

    @property
    def xyz_and_features(self):
        # get coordinates (XYZ) and features as numpy array
        if self.df_pc is None:
            self.load_data()
        xyz = self.xyz
        feat = self.features
        xyz_feat = np.hstack((xyz, feat))
        return xyz_feat

    @property
    def pointdensity(self):
        # calculate the point density as number of points divided by the area of the bounding box
        area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        return len(self) / area

    def __len__(self):
        return self.df_pc.shape[0]

    def __repr__(self):
        return f'Point cloud Dataset {self.file.split("/")[-1]}'

    def __str__(self):
        return f'Dataset {self.file.split("/")[-1]} with {len(self):,} points'


class kNNBatchDataset(Dataset):
    # This class can be used to create batches from a single point cloud
    def __init__(self, k, spacing, attributes=None, *args, **kwargs):
        super(kNNBatchDataset, self).__init__(*args, **kwargs)
        self.spacing = spacing  # distance between grid points
        self.k = k  # number of points for kNN
        self.tree = None    # initialize KDTree
        self.currIdx = 0    # current index
        # calculate number of rows and columns of the grid
        self.num_cols = (self.xmax - self.xmin - self.spacing / 2) // self.spacing + 1
        self.num_rows = (self.ymax - self.ymin - self.spacing / 2) // self.spacing + 1
        # calc number of batches, which is equal to number of grid points
        self.num_batches = int(self.num_cols * self.num_rows)
        self.rndzer = list(range(self.num_batches))
        np.random.shuffle(self.rndzer)

        # add point index if not available.
        # The point index ensures that the batch can be traced back to a point cloud after classification.
        if 'point_index' not in self.df_pc.columns:
            self.df_pc['point_index'] = np.arange(len(self.df_pc))
        # select defined attributes if defined
        if attributes:
            self.select_features(attributes)

        # build a KD-Tree for the point cloud
        self.buildKD()
        print('finished with building KD-Tree')

    def buildKD(self):
        # build a KD-Tree for the point cloud
        self.tree = KDTree(self.xyz[:, 0:2])  # build only on XY

    def getBatches(self, batch_size=None):
        # get the points of each batch
        # function returns two numpy arrays
        centers = []
        self.currIdx = 0

        if batch_size is None:
            batch_size = self.num_batches

        # calculate the centers of the batches
        for i in range(batch_size):
            if self.currIdx >= self.num_batches:
                break
            centers.append([self.xmin + self.spacing / 2 + (self.currIdx // self.num_cols) * self.spacing,
                            self.ymin + self.spacing / 2 + (self.currIdx % self.num_cols) * self.spacing])
            self.currIdx += 1

        if centers:
            # get the k nearest neighbors for every center
            _, idx = self.tree.query(centers, k=self.k)
            xyz_feats = self.df_pc.drop(['classification'], axis=1, errors='ignore').to_numpy()
            return xyz_feats[idx, :], self.labels[idx]
        else:
            return None, None

    def save_batches(self, path):
        # function to save the batches to a defined path

        # get the calculated batches
        points, labels = self.getBatches(self.num_batches)
        # define output path
        path_file = path + self.file.split("/")[-1].replace('.laz', '') + f'_bs-{self.k//1000}k/'

        # create folder, if not already there
        if not os.path.isdir(path_file):
            os.makedirs(path_file)

        # iterate throw every batch and save it
        for i in range(len(points)):
            # define export path
            ex_path = path_file + f'{self.name.replace(".laz","")}_batch_{i:03d}.las'
            arr = np.hstack((points[i, :, :], np.expand_dims(labels[i, :], axis=-1)))
            # create list with all attributes --> classification is the last one.
            cols = list(self.df_pc.drop('classification', axis=1).columns)
            cols.append('classification')
            save_df_to_las(pd.DataFrame(arr, columns=cols), ex_path) # --> Function from tools/lastools.py
            print(f'finished with batch nr {i} from {len(points)}')


class PointcloudDataset(data.Dataset):
    # Dataset Class for training of pointnet2 model
    # This Torch-Dataset Object returns only ALL attributes
    def __init__(self, file_path, augmentation=True, aug_prob=0.5):

        # initialize features
        self.lst_files = []  # list with all files within the path
        self.lst_pointclouds = []  # list with all point clouds
        self.total_files = len(os.listdir(file_path))  # number of total batches
        self.aug_prob = aug_prob  # Probability to do DataAugmentation
        self.do_augmentation = augmentation  # varibale defines if we want to use data augmentation

        # iterate throw files
        for i, file in enumerate(os.listdir(file_path)):
            if file.endswith('.las') or file.endswith('.laz'):
                path = file_path + '/' + file
                self.lst_files.append(path)
                # generate a dataset object and add the object to the list of point clouds
                feats = ['X', 'Y', 'Z', 'intensity', 'return_number', 'red', 'green', 'blue',
                         'hue', 'saturation', 'value', 'classification', 'point_index']
                dataset = Dataset(path).select_features(feats)

                self.lst_pointclouds.append(dataset)

    def __getitem__(self, index):
        # the returned item will be the returned item from the DataLoader
        dataset = self.lst_pointclouds[index]
        coords_feats = dataset.xyz_and_features

        # do data augmentation if 'do_augmentation' is True
        if self.do_augmentation:
            xyz = coords_feats[:, 0:3]
            hsv = coords_feats[:, 5:8]

            # data augmentation
            xyz = z_jittering(xyz, prob=0, stddev=0.05)
            xyz = rotate_random_around_z(xyz)
            hsv = hsv_jittering(hsv, prob=0)
            hsv = hsv_shift(hsv, prob=0)
            hsv = random_colordrop(hsv, prob=0.0)

            # add results to coordinates and features
            coords_feats[:, 0:3] = xyz
            coords_feats[:, 5:8] = hsv

        return torch.from_numpy(coords_feats), dataset.labels

    def __len__(self):
        return self.total_files


class PointcloudDataset_v2(data.Dataset):
    # Dataset Class for training of pointnet2 model
    # This Torch-Dataset Object returns only SIX attributes: X, Y, Z, Hue, Saturation, Value!!
    def __init__(self, file_path, augmentation=True, aug_prob=0.5):

        # initialize features
        self.lst_files = []   # list with all files within the path
        self.lst_pointclouds = []   # list with all point clouds
        self.total_files = len(os.listdir(file_path))   # number of total batches
        self.aug_prob = aug_prob    # Probability to do DataAugmentation
        self.do_augmentation = augmentation     # varibale defines if we want to use data augmentation

        # iterate throw files
        for i, file in enumerate(os.listdir(file_path)):
            if file.endswith('.las') or file.endswith('.laz'):
                path = file_path + '/' + file
                self.lst_files.append(path)
                # generate a dataset object and add the object to the list of point clouds

                feats = ['X', 'Y', 'Z', 'red',  'hue', 'saturation', 'value', 'green', 'blue', 'classification', 'point_index']
                dataset = Dataset(path).select_features(feats)

                self.lst_pointclouds.append(dataset)

    def __getitem__(self, index):
        # the returned item will be the returned item from the DataLoader
        dataset = self.lst_pointclouds[index]
        coords_feats = dataset.xyz_and_features

        # do data augmentation if 'do_augmentation' is True
        if self.do_augmentation:
            xyz = coords_feats[:, 0:3]
            hsv = coords_feats[:, 5:8]

            # data augmentation
            xyz = z_jittering(xyz, prob=0, stddev=0.05)
            xyz = rotate_random_around_z(xyz)
            hsv = hsv_jittering(hsv, prob=0)
            hsv = hsv_shift(hsv, prob=0)
            hsv = random_colordrop(hsv, prob=0.0)

            # add results to coordinates and features
            coords_feats[:, 0:3] = xyz
            coords_feats[:, 5:8] = hsv

        return torch.from_numpy(coords_feats), dataset.labels

    def __len__(self):
        return self.total_files


if __name__ == '__main__':
    path = '../batches/testing'
    dataset = PointcloudDataset(path)

    print(f'shape of featrues from batch #{5}:\n {dataset[5][0].shape}\n\ntotal number of batches: {len(dataset)}')
