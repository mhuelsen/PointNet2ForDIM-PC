import pandas as pd
import numpy as np
from Neighborhood.Distances import *
from skspatial.objects import Plane, Points


def center_of_mass(df, mode="3D"):
    '''
    calculate the center of mass of a given PointCloud
    :param df: PointCloud
    :type df: pandas.DataFrame
    :param mode: define mode - 3D or 2D
    :return: pandas.Series with the center of mass
    '''

    # Point - Count
    k = len(df)
    # calculate means
    x_com = df.X.sum() / k
    y_com = df.Y.sum() / k

    if mode == "3D":
        z_com = df.Z.sum() / k
        COM = pd.Series(data={'X': x_com, 'Y': y_com, 'Z': z_com}, index=['X', 'Y', 'Z'])

    elif mode == "2D":
        COM = pd.Series(data={'X': x_com, 'Y': y_com}, index=['X', 'Y'])

    else:
        COM = pd.Series()

    return COM


def build_structurtensor(df, mode='3D'):
    '''
    calculate the structure tensor of a point cloud
    :param df: Pointcloud
    :type df: pandas.DataFrame
    :param mode: define mode - '3D' or '2D'
    :type mode: str
    :return: numpy.Array with the structuretensor as Matrix
    '''

    # df = DatamFrame mit allen Punkten

    # Anzahl der Punkte

    if mode == '3D':
        # Schwerpunkt
        COM = center_of_mass(df)
        # Reduzierung der Koordinaten
        df_red = pd.DataFrame({'X': (df.X - COM.X), 'Y': (df.Y - COM.Y), 'Z': (df.Z - COM.Z)})

    elif mode == '2D':
        # Schwerpunkt
        COM = center_of_mass(df, mode=mode)

        # Reduzierung der Koordinaten
        df_red = pd.DataFrame({'X': (df.X - COM.X), 'Y': (df.Y - COM.Y)})

    # Convertierung der reduzierten Koordinaten in Numpy-Array
    coords = df_red.to_numpy()

    # Berechnung des Struktur-Tesors
    tensor = np.matmul(np.transpose(coords), coords)

    return tensor


def delta_Z(point, df):
    '''
    calculates the maximal difference between Z-coordinates
    :param point: Center point
    :type point: pandas.Series
    :param df: Pointcloud of surrounding points
    :type df: pandas.DataFrame
    :return: float with Maximum Delta Z
    '''

    df = pd.concat([df, point])

    return float(df.Z.max() - df.Z.min())


def std_Z(point, df):
    '''
    calculates the standard deviation between Z-coordinates
    :param point: Center point
    :type point: pandas.Series
    :param df: Pointcloud of surrounding points
    :type df: pandas.DataFrame
    :return: float with Std. Z
    '''

    df = pd.concat([df, point])

    return float(df.Z.std())


def radius_kNN(point, df):
    '''
    calculates the radius that defines the kNN-Neighborhood
    :param point: Center point
    :type point: pandas.Series
    :param df: Pointcloud of surrounding points
    :type df: pandas.DataFrame
    :return: float with the radius
    '''

    # calculate distances
    dist = calc_3D_dist(point, df.loc[:, 'X':'Z'])
    # calculate the max difference
    max_dist = dist.max()

    return float(max_dist)


def local_pointdensity_kNN(point, df):
    '''
    calculates the local pointdensity of a kNN Neighborhood
    :param point: Center point
    :type point: pandas.Series
    :param df: Pointcloud of surrounding points
    :type df: pandas.DataFrame
    :return: float with the pointdensity
    '''

    D = (len(df) + 1) / (4 / 3 * np.pi * radius_kNN(point, df) ** 3)

    return float(D)


def Verticality(df):
    '''
    calculates the verticality of a pointcloud
    :param df: Pointcloud of surrounding points
    :type df: pandas.DataFrame
    :return: float with the point density
    '''

    # Berechne Strukturtensor
    tensor = build_structurtensor(df)

    # Berechne Eigenwerte und Eigenvektoren
    w, v = np.linalg.eig(tensor)

    # Eigenvektor in Z-Richtung
    vz = v[:, 2]

    return 1 - np.linalg.norm(vz)


def roughness(point, df):
    points = Points(df.loc[:, 'X':'Z'].to_numpy())

    plane = Plane.best_fit(points)
    dist = plane.distance_point(point.loc['X':'Z'].to_numpy())

    return dist


def geom_3D_Features(point, df):
    '''
    calculates all the geometric 3D_Features
    :param point: Center point
    :type point: pandas.Series
    :param df: Pointcloud of surrounding points
    :type df: pandas.DataFrame
    :return: pandas.Series with alls geometric 3D Features
    '''
    geom_Feat = pd.Series(data={
        'Delta_Z': delta_Z(point, df),
        'Std_Z': std_Z(point, df),
        'Radius kNN': radius_kNN(point, df),
        'local Pointdensity': local_pointdensity_kNN(point, df),
        'Verticality': Verticality(df),
        'roughness': roughness(point, df)
    }, index=['Delta_Z', 'Std_Z', 'Radius kNN', 'local Pointdensity', 'Verticality', 'roughness'])

    return geom_Feat
