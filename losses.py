import torch
import torch.nn.functional as F


def dice_loss(y_pred,y_true,smooth=1e-5):
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims+2))
    intersection = 2 * (y_true * y_pred).sum(dim=vol_axes)
    union = y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)
    dice = (intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def smoothing_loss(deformation_field):
    dx = torch.abs(deformation_field[:, :, 1:, :, :] - deformation_field[:, :, :-1, :, :])
    dy = torch.abs(deformation_field[:, :, :, 1:, :] - deformation_field[:, :, :, :-1, :])
    dz = torch.abs(deformation_field[:, :, :, :, 1:] - deformation_field[:, :, :, :, :-1])
    
    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)# losses.py


def bending_energy_loss(flow):
    # Second-order derivatives for smoothness
    d2x = flow[:,:,2:] - 2*flow[:,:,1:-1] + flow[:,:,:-2]
    d2y = flow[:,:,:,2:] - 2*flow[:,:,:,1:-1] + flow[:,:,:,:-2]
    d2z = flow[:,:,:,:,2:] - 2*flow[:,:,:,:,1:-1] + flow[:,:,:,:,:-2]
    

    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)

def jacobian_det_loss(flow):
    """Prevent folding through Jacobian analysis"""
    J = torch.stack(torch.gradient(flow, dim=(2,3,4)), dim=2) 
    det = torch.det(J.permute(0,3,4,5,1,2))  
    return torch.mean(F.relu(-det))  

def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))

def composite_loss(pred, target, flow):
    return (
        0.8 * dice_loss(pred, target) + 
        0.2 * cross_entropy_loss(pred, target) +
        0.1 * bending_energy_loss(flow) +
        0.01 * jacobian_det_loss(flow)
    )

import torch
import torch.nn.functional as F

def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Chamfer Distance between two point clouds.
    
    Args:
        p1 (torch.Tensor): First point cloud, shape (B, N, 3)
        p2 (torch.Tensor): Second point cloud, shape (B, M, 3)
        
    Returns:
        torch.Tensor: The Chamfer distance.
    """
    dist1 = torch.cdist(p1, p2)
    dist1, _ = torch.min(dist1, dim=2)
    
    dist2 = torch.cdist(p2, p1)
    dist2, _ = torch.min(dist2, dim=2)
    
    return torch.mean(dist1) + torch.mean(dist2)

def laplacian_smoothing_loss(flow: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """
    Penalizes local rigidity in the deformation field.
    Finds k-nearest neighbors and encourages their flow vectors to be similar.
    
    Args:
        flow (torch.Tensor): The predicted flow, shape (B, N, 3)
        xyz (torch.Tensor): The original point coordinates, shape (B, N, 3)
        
    Returns:
        torch.Tensor: The smoothing loss.
    """
    k = 8
    # Find k-nearest neighbors
    dist = torch.cdist(xyz, xyz)
    _, knn_idx = torch.topk(dist, k, dim=-1, largest=False)
    
    # Index the flow to get neighbor flows
    B, N, _ = flow.shape
    batch_indices = torch.arange(B).view(B, 1, 1).to(flow.device)
    neighbor_flow = flow[batch_indices, knn_idx] # (B, N, k, 3)
    
    # Calculate the difference
    diff = flow.unsqueeze(2) - neighbor_flow # (B, N, k, 3)
    
    return torch.mean(torch.sum(diff ** 2, dim=-1))

def composite_loss_pointcloud(warped_xyz, fixed_xyz, flow, original_moving_xyz, smooth_weight=0.1):
    """
    Composite loss for point cloud registration.
    """
    match_loss = chamfer_distance(warped_xyz, fixed_xyz)
    smooth_loss = laplacian_smoothing_loss(flow, original_moving_xyz)
    
    return match_loss + smooth_weight * smooth_loss