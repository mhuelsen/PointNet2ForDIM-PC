# PointNet2ForDIM-PC
👨‍🎓 This code repository is part of Markus Hülsen's master thesis called: <br><pre>
  *Analysis of deep learning methods for the semantic segmentation of photogrammetric point clouds from aerial images*. </pre>
🤝 It was written in collaboration with the [Landesamt für Geoinformation und Landesvermessung Niedersachsen (LGLN)](https://www.lgln.niedersachsen.de/startseite/) and the [Jade University of Applied Sciences](https://www.jade-hs.de/) and was submitted on August 25, 2023.

## Why all this?
✈️📷 The aim of the work is the semantic segmentation of point clouds that were generated from aerial images using Structure from Motion and Dense Image Matching. To do this, an extensive data set is first annotated in order to obtain suitable training data. <br>
The ground truth data consists of four classes: 
- ground ⛰️
- buildings 🏠
- vegetation 🌳
- humanmade 🚗 and
- bridges 🌉

🖱️ The annotated data set can be downloaded at [AcademicCloud](https://sync.academiccloud.de/index.php/s/hj5C7ebHkkTZkvQ).<br>

🚀 The deep learning architecture PointNet++ is then applied, with the parameters used largely based on the work [*alsNet* of Lukas Winiwarter (2019)](https://github.com/lwiniwar/alsNet#readme). <br>
🔥 The chosen approach achieves an overall accuracy of **96.5%**.

## Requirements
These scipts are tested with the following libarays.

Some of the libarays are not common and need to be installed on a special way:
- [PyTorch](https://pytorch.org/get-started/locally/) (2.0.1) with [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html) (1.1.0)
- [laspy[laszip]](https://laspy.readthedocs.io/en/latest/installation.html) (2.4.1)
- [pointnet2_pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch) (Pointnet++ Pytorch from Erik Wijmans (2018) - last update July 30th, 2021)
  - This is especially important for code `pointnet_utils.py`

KD-Tree and Confusion Matrix are calculated by
- [scipy.spatial](https://scipy.org/install/) (1.11.1) 
- [sklearn.metrics](https://scikit-learn.org/stable/install.html) (1.3.1.)

And some other standard common python libarays
- [numpy](https://numpy.org/install/) (1.25.)
- [tqdm](https://pypi.org/project/tqdm/) (4.66.1)
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html) (3.8.0)
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html) (2.1.1)

## Structure and Description
This repository consists of two folders: `Jupyter Notebooks` and `Python Code`. <br> 
First, the `Jupyter-Notebooks` are used to optimize the ground-truth data and to test some important functions for the PointNet++ model<br>
Next up, the `Python Code` is used to create batches, to train and validate the model.

### Jupyter-Notebooks
The initially annotated point cloud contains too many incorrect assignments, so optimization is necessary. In addition, the "Not Ground" class is divided into the "Vegetation" and "Humanmade" classes.<br>
Description of the subfolders:
1. Calculate height above DEM ➡️ Generate a digital elevation model from a classified point cloud and calculate the height above ground as an additional scalar field
2. Seperate building ➡️ Seperates the vegetation inside of the building-class with the use of different treshholds
3. Intersection with ALKIS ➡️ Intersects a point cloud with two-dimensional *Road*-polygons and adds `insde_road` as an additional scalar field
4. KMeans1: ➡️ First K-Means clustering to cluster the class "Not Ground"
5. KMeans2: ➡️ Second K-Means clustering to cluster the rest of class "Not Ground"
6. KMeans, DBScan and RF ➡️ Cluster the remaining "Not-Ground"-Points with the geometry and radiometry and classify the resulting clusters using Random Forest
7. PointNet++ ➡️ Test of some functions which are import for PointNet++ like Batching, *Farthest Point Sampling* and *Ball Query*

### Python Code
1. `create_batches.py` ➡️ script will import a las or laz file creates the batches with a defined number of points
2. `train_pointnet2.py` ➡️ script will import the batches and initalize a pointnet2 model.
   -  If defined it will use a predefined model to further train this model.
4. `classify_batches.py` ➡️ This script will import the batches in a defined path to classify every batch.
   - after batch classification it will combine the batches back to a single pointcloud, in order to create a fully classified cloud
5. `validate_model.py` ➡️ script will import the batches from a defined path and uses the validation batch to calculate some statistics
   - In addition it will plot a confusion matrix and the ROC-plots

## Related Projects
- [Pointnet/Pointnet++ Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by Xu Yan (2019)
- [alsNet](https://github.com/lwiniwar/alsNet) by Lukas Winiwarter (2019)
- [pytorchpointnet++](https://github.com/erikwijmans/Pointnet2_PyTorch) by Erik Wijmans (2018)
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](http://stanford.edu/~rqi/pointnet2/) by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.
