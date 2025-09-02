import torch
import torch.nn.functional as F


def dice_loss(y_pred, y_true, smooth=1e-5):
    # Handle both one-hot and single-channel inputs
    if len(y_pred.shape) == 3:  # [D, H, W] - single channel
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        y_true = y_true.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # Now both should be [B, C, D, H, W] format
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims+2))

    intersection = (y_pred * y_true).sum(dim=vol_axes)
    union = y_pred.sum(dim=vol_axes) + y_true.sum(dim=vol_axes)

    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()

def smoothing_loss(deformation_field):
    dx = torch.abs(deformation_field[:, :, 1:, :, :] - deformation_field[:, :, :-1, :, :])
    dy = torch.abs(deformation_field[:, :, :, 1:, :] - deformation_field[:, :, :, :-1, :])
    dz = torch.abs(deformation_field[:, :, :, :, 1:] - deformation_field[:, :, :, :, :-1])
    
    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)

def bending_energy_loss(flow):
    flow = flow.unsqueeze(0)

    # Second-order derivatives for smoothness
    d2x = flow[:,:,2:] - 2*flow[:,:,1:-1] + flow[:,:,:-2]
    d2y = flow[:,:,:,2:] - 2*flow[:,:,:,1:-1] + flow[:,:,:,:-2]
    d2z = flow[:,:,:,:,2:] - 2*flow[:,:,:,:,1:-1] + flow[:,:,:,:,:-2]
    
    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)

def jacobian_det_loss(flow):
    """Prevent folding through Jacobian analysis"""
    flow = flow.unsqueeze(0)
    J = torch.stack(torch.gradient(flow, dim=(2,3,4)), dim=2) 
    det = torch.det(J.permute(0,3,4,5,1,2))  
    return torch.mean(F.relu(-det))  

def cross_entropy_loss(pred, target):
    # Handle single-channel inputs by adding channel dimension
    if len(pred.shape) == 3:  # [D, H, W]
        pred = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        target = target.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # For binary segmentation, we need to create a 2-class problem
    # pred: [B, 1, D, H, W] -> [B, 2, D, H, W]
    # target: [B, 1, D, H, W] -> [B, D, H, W] (long tensor)
    
    # Convert target to long tensor for cross entropy
    target_long = target.squeeze(1).long()
    
    # For binary case, we can use BCE with logits or create 2-class output
    # Here we'll use BCE with logits for simplicity
    return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.squeeze(1))

def composite_loss(pred, target, flow):
    # Ensure flow is in the right format for loss calculations
    if len(flow.shape) == 5:  # [B, 3, D, H, W]
        flow_for_loss = flow.squeeze(0)  # [3, D, H, W]
    else:
        flow_for_loss = flow
    
    return (
        0.8 * dice_loss(pred, target) + 
        0.2 * cross_entropy_loss(pred, target) +
        0.1 * bending_energy_loss(flow_for_loss) +
        0.01 * jacobian_det_loss(flow_for_loss)
    )
