import torch
import torch.nn.functional as F


def dice_loss(probs, targets, smooth=1e-5):
    # probs: [B, C, D, H, W] after softmax; targets: [B, C, D, H, W] one-hot
    if probs.dim() == 4:
        probs = probs.unsqueeze(0)
    if targets.dim() == 4:
        targets = targets.unsqueeze(0)
    ndims = probs.dim() - 2
    reduce_axes = list(range(2, ndims + 2))
    intersection = (probs * targets).sum(dim=reduce_axes)
    union = probs.sum(dim=reduce_axes) + targets.sum(dim=reduce_axes)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def smoothing_loss(deformation_field):
    dx = torch.abs(deformation_field[:, :, 1:, :, :] - deformation_field[:, :, :-1, :, :])
    dy = torch.abs(deformation_field[:, :, :, 1:, :] - deformation_field[:, :, :, :-1, :])
    dz = torch.abs(deformation_field[:, :, :, :, 1:] - deformation_field[:, :, :, :, :-1])
    
    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)

def bending_energy_loss(flow):
    # flow: [B, 3, D, H, W]
    if flow.dim() == 4:
        flow = flow.unsqueeze(0)
    d2x = flow[:, :, 2:] - 2 * flow[:, :, 1:-1] + flow[:, :, :-2]
    d2y = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    d2z = flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]
    return (d2x.pow(2).mean() + d2y.pow(2).mean() + d2z.pow(2).mean())

def jacobian_det_loss(flow):
    # Disable for stability initially; can be re-enabled later with robust impl
    return torch.tensor(0.0, device=flow.device, dtype=flow.dtype)

def nll_from_probs(probs, targets, eps: float = 1e-6):
    # probs: [B, C, D, H, W], targets: [B, C, D, H, W] one-hot
    if probs.dim() == 4:
        probs = probs.unsqueeze(0)
    if targets.dim() == 4:
        targets = targets.unsqueeze(0)
    log_probs = torch.log(torch.clamp(probs, eps, 1.0))
    nll = -(targets * log_probs).sum(dim=1)  # [B, D, H, W]
    return nll.mean()

def composite_loss(probs, target_onehot, flow):
    # probs: [B, C, D, H, W]; target_onehot: [B, C, D, H, W]; flow: [B, 3, D, H, W]
    ce = nll_from_probs(probs, target_onehot)
    dice = dice_loss(probs, target_onehot)
    be = bending_energy_loss(flow)
    jac = jacobian_det_loss(flow)
    return 0.7 * ce + 0.3 * dice + 0.1 * be + 0.0 * jac
