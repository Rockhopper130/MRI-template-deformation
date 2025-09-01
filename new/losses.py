import torch
import torch.nn.functional as F


def dice_loss(y_pred,y_true,smooth=1e-5):
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims+2))
    intersection = 2 * (y_true * y_pred).sum(dim=vol_axes)
    union = y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)
    dice = (intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def dice_loss(y_pred, y_true, smooth=1e-5):
    # Assumes y_pred and y_true are of shape [N, C, D, H, W]
    ndims = len(y_pred.shape)
    # Sum over spatial dimensions D, H, W
    vol_axes = list(range(2, ndims)) 

    intersection = (y_pred * y_true).sum(dim=vol_axes)
    union = y_pred.sum(dim=vol_axes) + y_true.sum(dim=vol_axes)

    # Note: We add 2* here because we are calculating the intersection
    # in the numerator.
    dice_score = (2. * intersection + smooth) / (union + smooth)

    # Return the loss, which is 1 - Dice score, averaged over batch and channels
    return 1 - dice_score.mean()

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
