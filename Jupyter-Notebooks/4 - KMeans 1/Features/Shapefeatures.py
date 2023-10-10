import numpy as np
from Features.Geometric_Features import *


def Linearity(eigvals):
    '''
    calculates the linearity
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the linearity
    '''

    return float((eigvals[0] - eigvals[1]) / eigvals[0])


def Planarity(eigvals):
    '''
    calculates the planarity
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the planarity
    '''

    return float((eigvals[1] - eigvals[2]) / eigvals[0])


def Scattering(eigvals):
    '''
    calculates the scattering
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the scattering
    '''
    return eigvals[2] / eigvals[0]


def Omnivariance(eigvals):
    '''
    calculates the omnivariance
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the omnivariance
    '''

    return np.prod(eigvals) ** (1 / len(eigvals))


def Ansiotropy(eigvals):
    '''
    calculates the ansiotropy
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the ansiotropy
    '''
    return float((eigvals[0] - eigvals[2]) / eigvals[0])


def Eigenentropy(eigvals):
    '''
    calculates the eigenentropy
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the eigenentropy
    '''

    sum_e = 0
    for eigval in eigvals:
        sum_e += eigval * np.log(eigval)

    return -sum_e


def Sum_Eigenvals(eigvals):
    '''
    calculates the sum of the eigenvalues
    :param eigvals: Array with eigenvalues
    :type eigvals: numpy.array
    :return: float with the Sum of the eigenvals
    '''
    return np.sum(eigvals)


def Change_of_Curvature(eigvals):
    '''
    calculates the change of the curvature
    :param eigvals: Array with normalized eigenvalues
    :type eigvals: numpy.array
    :return: float with the change of the curvature
    '''

    return float(eigvals[2] / Sum_Eigenvals(eigvals))


def geom_3D_Shapeproperties(df):
    '''
    calculates all the 3D Shape Features
    :param df: PointCloud with DataFrame
    :type df: pandas.DataFrame
    :return: pandas.Series with all Shape features
    '''

    # calculate structuretensor
    tensor = build_structurtensor(df)

    # calculate eigenvalues
    eigvals = np.linalg.eigvals(tensor)

    # catch error, if eigenvalue is less than or equal to zero
    eigvals[eigvals <= 0] = 10 ** -20

    # normalisation of the eigenvalues
    norm_eigvals = eigvals / np.sum(eigvals)

    # caltulate the shapefeatures
    geom_Feat = pd.Series(data={
        'Linearity': Linearity(norm_eigvals),
        'Planarity': Planarity(norm_eigvals),
        'Scattering': Scattering(norm_eigvals),
        'Omnivariance': Omnivariance(norm_eigvals),
        'Ansiotropy': Ansiotropy(norm_eigvals),
        'Eigenentropy': Eigenentropy(norm_eigvals),
        'Sum_Eigenvals': Sum_Eigenvals(eigvals),
        'Change_of_Curvature': Change_of_Curvature(norm_eigvals)
    }, index=['Linearity', 'Planarity', 'Scattering', 'Omnivariance',
              'Ansiotropy', 'Eigenentropy', 'Sum_Eigenvals', 'Change_of_Curvature'])

    return geom_Feat
