import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# The pointnet2_ops libary is published by Erik Wijamns (2018) and available at:
#        https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops import pointnet2_utils as utils

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script defines different utility for the PoinNet++ Model and the Model itself
some parts of the script are published by Xu Yan (2019): https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""


def sample_and_group(npoint: int, radius: float, nsample: int,
                     xyz: torch.Tensor, points: torch.Tensor, returnfps: bool = False):
    """
    Function to sample points with FPS and group them with a ball query
    Input:
        npoint: Number of points that is sampled from FPS
        radius: Radius for the ball query
        nsample: max. number of neighboring points for ball query
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    # Batchsize (B), num of points (N), num of channels (C)
    B, N, C = xyz.shape
    # number of Sampling point
    S = npoint

    xyz = xyz.float().contiguous().to(torch.device('cuda'))
    points = points.to(torch.device('cuda'))

    # get indices farthest point sampling
    fps_idx = utils.furthest_point_sample(xyz, npoint)  # [B, npoint, C]

    xyz_flipped = xyz.transpose(1, 2).float().contiguous()
    # gather points with indices
    new_xyz = utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
    # get indices fixed radius ball query
    idx = utils.ball_query(radius, nsample, xyz, new_xyz)
    # gather coords of points with indices
    grouped_xyz = utils.grouping_operation(xyz_flipped, idx).permute(0, 2, 3, 1)  # [B, npoint, nsample, C]
    # center the groups with the FPS-point
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    # add features if available
    if points is not None:
        points_flipped = points.transpose(1, 2).contiguous().float()
        grouped_points = utils.grouping_operation(points_flipped, idx).permute(0, 2, 3, 1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    # return points and features
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list):
        """
        Set Abstraction Layer for the PointNet2 Modul
        :param npoint: number of points for FPS
        :param radius: radius of the fixed ball query
        :param nsample: numbers of points in the fixed ball query
        :param in_channel: number of input channels of the Layer
        :param mlp: List with the number of filters for the 2D-Convolution
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        # empty list with convolutions
        self.mlp_convs = nn.ModuleList()
        # empty list with batch normalisation
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        # generate  MLPs
        for out_channel in mlp:
            # append convolution to list
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            # append batch normalisation to list
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input features data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # sample points with FPS and group them with fixed ball query
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1).float()    # [B, C+D, nsample, npoint]

        # Do the 2D Convolution with Kernelsize = 1 and do batch normalisation
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list):
        """
        Feature Propagation Layer
        :param in_channel: number of input channels for the MLPs
        :param mlp: List with the number of filters for MLPs
        """
        super(PointNetFeaturePropagation, self).__init__()
        # Modul list with Convolutions and batch normalisations
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        # iterate throw the Layers of the MLP
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, unknown_xyz: torch.Tensor, known_xyz: torch.Tensor,
                unknown_points: torch.Tensor, known_points: torch.Tensor):
        """
        Input:
            unknown_xyz: input points position data, [B, N, 3]
            known_xyz: sampled input points position data, [B, S, 3]
            unknown_points: input points data, [B, N, D]
            known_points: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, N, D']
        """
        unknown_points = unknown_points.permute(0, 2, 1).to(torch.device('cuda'))
        known_points = known_points.permute(0, 2, 1).to(torch.device('cuda'))
        unknown_xyz = unknown_xyz.to(torch.device('cuda'))
        known_xyz = known_xyz.to(torch.device('cuda'))

        if known_xyz is not None:
            # calculate distance and index of the three nearest points
            dists, idx = utils.three_nn(unknown_xyz, known_xyz)
            # invert the distances
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # normalize distances to get the weights
            weight = dist_recip / norm
            # interpolate the features
            interpolated_points = utils.three_interpolate(known_points, idx, weight).permute(0, 2, 1)
        else:
            interpolated_points = known_points.expand(
                *(known_points.size[0:2] + [unknown_xyz.size(1)])
            ).permute(0, 2, 1)

        if unknown_points is not None:
            # concat results
            new_points = torch.cat([unknown_points.permute(0, 2, 1), interpolated_points], dim=-1)
        else:
            new_points = interpolated_points.permute(0, 2, 1)

        new_points = new_points.permute(0, 2, 1)

        # shift new to cpu, when model is on cpu
        # new_points = new_points.to(torch.device('cpu'))

        # iterate throw layers of MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points.float())))

        return new_points.permute(0, 2, 1).to(torch.device('cuda'))


class PointNet2(nn.Module):
    """STANDARD POINTNET++ MODEL WITH ALL 8 ATTRIBUTES"""
    def __init__(self, num_classes: int):
        super(PointNet2, self).__init__()

        # set abstraction layers                                               in_channel=5+3
        self.sa1 = PointNetSetAbstraction(npoint=8192, radius=1.0, nsample=16, in_channel=5+3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=4096, radius=5.0, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=2048, radius=15.0, nsample=64, in_channel=256 + 3, mlp=[128, 128, 256])

        # feature propagation
        self.fp3 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+5, mlp=[128, 64])
        #                                     in_channel=256+5
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz):
        # xyz = xyz.permute(0, 2, 1)
        l0_points = xyz[:, :, 3:].contiguous().float()
        l0_xyz = xyz[:, :, 0:3].contiguous().float()

        # set abstraction
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # feature propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points).contiguous().float()
        l0_points = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class PointNet2_v2(nn.Module):
    """ THIS POINTNET++ MODEL TAKES 6 ATTRIBUTES AS INPUTS - XYZ + HSV"""
    def __init__(self, num_classes: int):
        super(PointNet2_v2, self).__init__()

        # set abstraction layers                                               in_channel=5+3
        self.sa1 = PointNetSetAbstraction(npoint=8192, radius=1.0, nsample=16, in_channel=3+3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=4096, radius=5.0, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=2048, radius=15.0, nsample=64, in_channel=256 + 3, mlp=[128, 128, 256])

        # feature propagation
        self.fp3 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+3, mlp=[128, 64])
        #                                     in_channel=256+5
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz):
        # xyz = xyz.permute(0, 2, 1)
        l0_points = xyz[:, :, 3:].contiguous().float()
        l0_xyz = xyz[:, :, 0:3].contiguous().float()

        # set abstraction
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # feature propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points).contiguous().float()
        # l0_points = l0_points.to(torch.device('cpu'))
        l0_points = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        # x = F.softmax(x, dim=2)
        return x


class PointNet2_v3(nn.Module):
    """ THIS POINTNET++ MODEL HAS 4 SA-LAYERS"""
    def __init__(self, num_classes):
        super(PointNet2_v3, self).__init__()

        # set abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=16384, radius=1.0, nsample=16, in_channel=5+3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=8192, radius=2.0, nsample=16, in_channel=128+3, mlp=[64, 64, 128])
        self.sa3 = PointNetSetAbstraction(npoint=4096, radius=5.0, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa4 = PointNetSetAbstraction(npoint=2048, radius=15.0, nsample=64, in_channel=256 + 3, mlp=[128, 128, 256])

        # feature propagation
        self.fp4 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+5, mlp=[128, 64])

        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz):
        # xyz = xyz.permute(0, 2, 1)
        l0_points = xyz[:, :, 3:].contiguous().float()
        l0_xyz = xyz[:, :, 0:3].contiguous().float()

        # set abstraction
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # feature propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points).contiguous().float()
        # l0_points = l0_points.to(torch.device('cpu'))
        l0_points = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        # x = F.softmax(x, dim=2)
        return x


# from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
