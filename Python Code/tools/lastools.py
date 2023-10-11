import numpy as np
import laspy
import os
import pandas as pd

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script defines function to handle with las files
"""


def import_las_to_Dataframe(path):
    with laspy.open(path) as f:
        las = f.read()

    # read coordinates from las
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z}, index=np.arange(len(x)))

    for i in range(3, len(las.point_format.dimensions)):
        dim = las.point_format.dimensions[i].name
        try:
            df[dim] = np.array(las[dim])
        except:
            pass

    return df


def save_df_to_las(df, path):
    # function to save a DataFrame as a las or laz file
    header = laspy.LasHeader(point_format=3, version="1.2")

    # get the attributes needed for a las file
    atts = []
    for dim in header.point_format.dimensions:
        atts.append(dim.name)

    # if we have additional attributes (like Hue, Saturation, Value) we need to add an extra dimension
    for dim in df.columns:
        if dim not in atts:
            header.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))

    las_new = laspy.LasData(header)

    # save X, Y and Z to las
    las_new.x = df.X.to_numpy()
    las_new.y = df.Y.to_numpy()
    las_new.z = df.Z.to_numpy()

    for col in df.loc[:, 'intensity':].columns:
        las_new[col] = df[col].to_numpy()

    las_new.write(path)


def merge_pointclouds(input_path, save_path):
    # save files that are in laz-format
    lst_files = []
    for file in os.listdir(input_path):
        if file.endswith('.las') or file.endswith('.laz'):
            lst_files.append(input_path + '/' + file)

    lst_files = sorted(lst_files)

    df_all = pd.DataFrame()
    for i, file in enumerate(lst_files):
        df_sub = import_las_to_Dataframe(file)
        df_all = df_all.append(df_sub, ignore_index=True)
        print('added file number', i+1)
    save_df_to_las(df_all, save_path)
    print('saved pointcloud to', save_path)


if __name__ == '__main__':
    # path where the data ist stored
    data_path = '../../Daten/Datensatz_H3D/DIM_2022/7 - DBScan/edited'
    export_path = '../../Daten/Datensatz_H3D/DIM_2022/7 - DBScan/edited/merged_pointcloud.laz'
    merge_pointclouds(data_path, export_path)


