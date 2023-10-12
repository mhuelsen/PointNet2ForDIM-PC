# PointNet2ForDIM-PC
👨‍🎓 This code repository is part of Markus Hülsen's master thesis called: <br><pre>
  *Analysis of deep learning methods for the semantic segmentation of photogrammetric point clouds from aerial images*. </pre>
🤝 It was written in collaboration with the [Landesamt für Geoinformation und Landesvermessung Niedersachsen (LGLN)](https://www.lgln.niedersachsen.de/startseite/) and the [Jade University of Applied Sciences](https://www.jade-hs.de/) and was submitted on August 25, 2023.

## Why all this?
✈️📷 The aim of the work is the semantic segmentation of point clouds that were generated from aerial images using Structure from Motion and Dense Image Matching. To do this, an extensive data set is first annotated in order to obtain suitable training data. <br>
The ground truth data consists of four classes: 
- ground
- buildings
- vegetation
- humanmade and
- bridges<br>
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
