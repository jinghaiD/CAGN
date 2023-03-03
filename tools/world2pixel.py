import pickle
import torch
import numpy as np
import copy
import argparse

def w2p(data, name):

    if name == 'hotel':
        homography_matrix = np.array(
            [
                [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]
            ]
        )
    elif name == 'eth':
        homography_matrix = np.array(
            [
                [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]
            ]
        )
    elif name == 'zara1':
        homography_matrix = np.array(
            [
                [0.02104651, 0, 0],
                [0, -0.02386598, 13.74680446],
                [0, 0, 1.0000000e+00]
            ]
        )
    elif name == 'zara2':
        homography_matrix = np.array(
            [
                [0.02104651, 0, 0],
                [0, -0.02386598, 13.74680446],
                [0, 0, 1.0000000e+00]
            ]
        )
    elif name == 'univ':
        homography_matrix = np.array(
            [
                [0.02104651, 0, 0],
                [0, -0.02386598, 13.74680446],
                [0, 0, 1.0000000e+00]
            ]
        )    
    homo_inverse = np.linalg.inv(homography_matrix)
    peds = data.shape[0]
    length = data.shape[1]
    z = np.ones((peds, length, 1))
    position = np.concatenate((data, z), axis=2)
    po = []
    for i in range(peds):
        abs = []
        for j in range(length):
            tmp = np.transpose(position[i][j])
            vis = np.dot(homo_inverse, tmp)
            w = vis[2]
            x_y = vis[:-1]
            x_y = x_y / w
            if name == 'eth' or name == 'hotel':
                x = x_y[1]
                y = x_y[0]
            else:
                x = x_y[0]
                y = x_y[1]
            abs.append([x,y])
        po.append(copy.deepcopy(abs))
    return np.asarray(po)
