import torch
from losses import chamfer_distance_loss, hausdorff_distance_loss, surface_loss, bending_energy_loss

# This is to experiment with different variants of loss functions

def compound_loss(pred, target, deformation_field, loss_weights):
    """
    Computes a weighted sum of different loss functions.

    Args:
        pred: Predicted segmentation or deformation field.
        target: Ground truth segmentation or deformation field.
        deformation_field: The predicted deformation field.
        loss_weights: Dictionary of weights for each loss function.
    
    Returns:
        total_loss: Weighted sum of selected losses.
        loss_dict: Individual loss values.
    """
    loss_dict = {}

    # Compute individual losses
    loss_dict["chamfer"] = chamfer_distance_loss(pred, target)
    loss_dict["hausdorff"] = hausdorff_distance_loss(pred, target)
    loss_dict["surface"] = surface_loss(pred, target)
    loss_dict["bending_energy"] = bending_energy_loss(deformation_field)

    # Compute weighted sum
    total_loss = sum(loss_weights[k] * loss_dict[k] for k in loss_weights if k in loss_dict)

    return total_loss, loss_dict
