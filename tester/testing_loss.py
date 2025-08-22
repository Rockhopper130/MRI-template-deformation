import torch
from losses import chamfer_distance_loss, hausdorff_distance_loss, surface_loss, bending_energy_loss

def test_loss_functions():
    print("Testing loss functions...")

    # Simple synthetic test cases
    pred = torch.rand((1, 3, 32, 32, 32))  # Random deformation field, we need to replace this section with actual MRI data soon
    target = torch.rand((1, 3, 32, 32, 32))

    deformation_field = torch.rand((1, 3, 32, 32, 32))

    chamfer_loss = chamfer_distance_loss(pred, target)
    hausdorff_loss = hausdorff_distance_loss(pred, target)
    surf_loss = surface_loss(pred, target)
    bend_loss = bending_energy_loss(deformation_field)

    print(f"Chamfer Distance Loss: {chamfer_loss.item():.6f}")
    print(f"Hausdorff Distance Loss: {hausdorff_loss.item():.6f}")
    print(f"Surface Loss: {surf_loss.item():.6f}")
    print(f"Bending Energy Loss: {bend_loss.item():.6f}")
    # More comprehensive testing is needed, also we can keep adding other loss functions here as we go
    assert chamfer_loss >= 0, "Chamfer loss should be non-negative"
    assert hausdorff_loss >= 0, "Hausdorff loss should be non-negative"
    assert surf_loss >= 0, "Surface loss should be non-negative"
    assert bend_loss >= 0, "Bending energy loss should be non-negative"

    print("All tests passed!")

if __name__ == "__main__":
    test_loss_functions()
