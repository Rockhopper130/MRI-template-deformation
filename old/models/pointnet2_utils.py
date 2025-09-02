import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Uses iterative farthest point sampling to select a subset of points.
    
    Args:
        xyz (torch.Tensor): Point cloud, shape (B, N, 3)
        npoint (int): Number of points to sample
    
    Returns:
        torch.Tensor: Indices of sampled points, shape (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Finds all points within a given radius for each query point.
    
    Args:
        radius (float): The radius of the ball.
        nsample (int): Max number of points to sample in each ball.
        xyz (torch.Tensor): All points, shape (B, N, 3).
        new_xyz (torch.Tensor): Query points, shape (B, S, 3).
        
    Returns:
        torch.Tensor: Indices of points within the ball, shape (B, S, nsample).
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Indexes points using the provided indices.
    
    Args:
        points (torch.Tensor): Input points tensor, shape (B, N, C).
        idx (torch.Tensor): Sample indices, shape (B, S) or (B, S, K).
        
    Returns:
        torch.Tensor: Indexed points, shape (B, S, C) or (B, S, K, C).
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction Layer.
    """
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int], group_all: bool):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        """
        Args:
            xyz (torch.Tensor): Input points coordinates, shape (B, N, 3).
            points (torch.Tensor): Input features, shape (B, N, D).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - new_xyz: Sampled points coordinates, shape (B, S, 3).
                - new_points: Sampled points features, shape (B, S, D').
        """
        xyz = xyz.contiguous()
        points = points.contiguous()

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3, device=xyz.device)
            grouped_points = points.view(xyz.shape[0], 1, xyz.shape[1], -1)
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # [B, npoint, nsample, C+D]
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points