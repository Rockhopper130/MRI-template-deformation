import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, index_points

class PointNet2SceneFlow(nn.Module):
    """
    Simplified PointNet++ based model for Scene Flow (Deformation) Estimation.
    This architecture is inspired by FlowNet3D.
    """
    def __init__(self, feature_channels=6):
        super().__init__()
        
        # Encoders for each point cloud
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.5, nsample=16, in_channel=feature_channels, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256], group_all=False)
        
        # Feature Propagation / Decoder layers
        self.fp3 = self._make_fp_module(256, [256, 256], 128)
        self.fp2 = self._make_fp_module(128, [128, 128], 64)
        self.fp1 = self._make_fp_module(64, [64, 64, 64], 64)

        # Final prediction heads
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 3, 1) # Output 3 values for flow (dx, dy, dz)

    def _make_fp_module(self, in_channels, mlp_channels, skip_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, mlp_channels[0], 1),
            nn.BatchNorm1d(mlp_channels[0]),
            nn.ReLU(),
            nn.Conv1d(mlp_channels[0], mlp_channels[1], 1),
            nn.BatchNorm1d(mlp_channels[1]),
            nn.ReLU(),
        )

    def _propagate_features(self, xyz1, xyz2, points1, points2, fp_module):
        """
        Feature propagation logic from PointNet++.
        """
        dist, idx = self._three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        new_points = fp_module(new_points)
        return new_points

    def _three_nn(self, xyz1, xyz2):
        """
        Finds the 3 nearest neighbors of xyz1 in xyz2.
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        dist = torch.cdist(xyz1, xyz2)
        dist, idx = torch.topk(dist, 3, dim=-1, largest=False)
        return dist, idx

    def forward(self, moving_pc, fixed_pc):
        """
        Args:
            moving_pc (torch.Tensor): The moving point cloud, shape (B, N, C)
            fixed_pc (torch.Tensor): The fixed point cloud, shape (B, N, C)
        
        Returns:
            torch.Tensor: The predicted flow (deformation), shape (B, N, 3)
        """
        # Separate coordinates and features
        l0_xyz_m = moving_pc[:, :, :3]
        l0_points_m = moving_pc[:, :, 3:]
        l0_xyz_f = fixed_pc[:, :, :3]
        l0_points_f = fixed_pc[:, :, 3:]

        # === Moving Point Cloud Encoder ===
        l1_xyz_m, l1_points_m = self.sa1(l0_xyz_m, l0_points_m)
        l2_xyz_m, l2_points_m = self.sa2(l1_xyz_m, l1_points_m)
        l3_xyz_m, l3_points_m = self.sa3(l2_xyz_m, l2_points_m)
        
        # === Fixed Point Cloud Encoder (shared weights) ===
        l1_xyz_f, l1_points_f = self.sa1(l0_xyz_f, l0_points_f)
        l2_xyz_f, l2_points_f = self.sa2(l1_xyz_f, l1_points_f)
        l3_xyz_f, l3_points_f = self.sa3(l2_xyz_f, l2_points_f)

        # === Flow Embedding (simple concatenation) ===
        # A more advanced model like FlowNet3D would have a specific layer here.
        # We will do the feature propagation on the moving cloud, using features
        # from the fixed cloud to inform the flow.
        
        # === Decoder ===
        # Propagate from l3 to l2
        l2_points_m_flow = self._propagate_features(l2_xyz_m, l3_xyz_m, l2_points_m, l3_points_m, self.fp3)
        # Propagate from l2 to l1
        l1_points_m_flow = self._propagate_features(l1_xyz_m, l2_xyz_m, l1_points_m, l2_points_m_flow.permute(0,2,1), self.fp2)
        # Propagate from l1 to l0
        l0_points_m_flow = self._propagate_features(l0_xyz_m, l1_xyz_m, l0_points_m, l1_points_m_flow.permute(0,2,1), self.fp1)

        # === Prediction Head ===
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points_m_flow))))
        flow = self.conv2(x)
        
        return flow.permute(0, 2, 1) # (B, N, 3)