# PointNet2ForDIM-PC
This code repository is part of Markus Hülsen's master thesis called *Analysis of deep learning methods for the semantic segmentation of photogrammetric point clouds from aerial images*. <br>
It was written in collaboration with the [Landesamt für Geoinformation und Landesvermessung Niedersachsen (LGLN)](https://www.lgln.niedersachsen.de/startseite/) and the [Jade University of Applied Sciences](https://www.jade-hs.de/) and was submitted on August 25, 2023.

The aim of the work is the semantic segmentation of point clouds that were generated from aerial images using Structure from Motion and Dense Image Matching. To do this, an extensive data set is first annotated in order to obtain suitable training data. The ground truth data consists of four classes: soil, buildings, vegetation, man-made and bridges. The annotated data set can be downloaded at [AcademicCloud](https://sync.academiccloud.de/index.php/s/hj5C7ebHkkTZkvQ)
The deep learning architecture PointNet++ is then applied, with the parameters used largely based on the work of Winiwarter (2019). The chosen approach achieves an overall accuracy of 96.5%.
