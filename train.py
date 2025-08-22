# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.field_models import UNet, PointNetDenseFlow
# from models.transform_models import SpatialTransformer
# from get_data import SegDataset
# from losses import dice_loss, smoothing_loss, composite_loss
# from tqdm import tqdm
# import numpy as np
# import argparse
# import os
# import wandb
# from torch.cuda.amp import GradScaler

# torch.cuda.empty_cache()

# # Logging
# wandb.init(project='seg-deformation')

# # def train(model, stn, dataloader, scaler, scheduler, optimizer, device, epoch, max_epochs=50, dice_weight=1.0, smooth_weight=0.1):
# #     model.train()
# #     total_loss = 0

# #     for moving, fixed in tqdm(dataloader, desc=f'Training for epoch: {epoch+1}/{max_epochs}', leave=False):
# #         moving = moving.to(device)
# #         fixed = fixed.to(device)

# #         optimizer.zero_grad()
        
# #         input_ = torch.cat([moving, fixed], dim=1)  # Shape: (B, 10, 128, 128, 128)

# #         deformation_field = model(input_)
# #         warped = stn(moving, deformation_field)

# #         loss = composite_loss(warped, fixed, deformation_field)

# #         scaler.scale(loss).backward()
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
# #         scaler.step(optimizer)
# #         scaler.update()
# #         scheduler.step()

# #         total_loss += loss.item()
# #     return total_loss / len(dataloader)

# def train(model, stn, dataloader, scaler, scheduler, optimizer, device, epoch, max_epochs=50):
#     model.train()
#     total_loss = 0

#     for moving, fixed in tqdm(dataloader, desc=f'Training for epoch: {epoch+1}/{max_epochs}', leave=False):
#         moving = moving.to(device) # Shape: (B, 4, D, H, W)
#         fixed = fixed.to(device)   # Shape: (B, 4, D, H, W)

#         optimizer.zero_grad(set_to_none=True)
        
#         # Concatenate all 8 channels (4 from moving, 4 from fixed)
#         input_ = torch.cat([moving, fixed], dim=1)

#         deformation_field = model(input_)
        
#         # FIX: Only warp the first channel (the binary mask), not the coordinate channels.
#         moving_mask = moving[:, :1, ...] # Shape: (B, 1, D, H, W)
#         fixed_mask = fixed[:, :1, ...]   # Shape: (B, 1, D, H, W)
        
#         warped_mask = stn(moving_mask, deformation_field)

#         # Loss should be calculated between the warped mask and the fixed mask.
#         loss = composite_loss(warped_mask, fixed_mask, deformation_field)

#         scaler.scale(loss).backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_txt', type=str, default='/tmp/oasis_data/train.txt', help='Path to the training file listing subject paths')
#     parser.add_argument('--template_path', type=str, default='/tmp/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy', help='Path to the template segmentation map')
#     parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
#     parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
#     parser.add_argument('--save_model_path', type=str, default='/tmp/oasis_data/weights_spherical/', help='Path to save the trained model')
#     args = parser.parse_args()

#     device = 'cuda:4'  # Change accordingly
#     # device = 'cpu'  # Change accordingly

#     print("Loading dataset")
#     train_dataset = SegDataset(args.train_txt, args.template_path)
#     print("Temp")
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
#     print("Dataset loaded.")

#     scaler = GradScaler()

#     model = UNet(in_channels=8, out_channels=3).to(device)
#     # model = PointNetDenseFlow(output_size=(128, 128, 128)).to(device)
#     stn = SpatialTransformer(size=(128, 128, 128), device=device).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=args.epochs * len(train_loader),
#         eta_min=1e-5
#     )

#     for epoch in range(args.epochs):
#         avg_loss = train(
#             model, stn, train_loader, scaler, scheduler, optimizer, device, epoch, args.epochs
#         )

#         print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")
#         wandb.log({'epoch': epoch + 1, 'loss': avg_loss})

#         if (epoch + 1) % 20 == 0:
#             model_path = f'{args.save_model_path}_epoch_{epoch + 1}.pth'
#             torch.save(model.state_dict(), model_path)
#             print(f"Model saved to {model_path}")

#     torch.save(model.state_dict(), args.save_model_path)
#     print(f"Model saved to {args.save_model_path}")

# if __name__ == "__main__":
#     main()
    
    
# SCORE TO BEAT
# Epoch 80/100, Loss: 0.2983 


import torch
from torch.utils.data import DataLoader
from models.pointnet_flow import PointNet2SceneFlow
from get_data import SegDataset
from losses import composite_loss_pointcloud
from tqdm import tqdm
import argparse
import wandb
from torch.cuda.amp import GradScaler, autocast

torch.cuda.empty_cache()

# Logging
wandb.init(project='seg-deformation-pointnet')

def train(model, dataloader, scaler, scheduler, optimizer, device, epoch, max_epochs=100, smooth_weight=0.1):
    model.train()
    total_loss = 0

    for moving_pc, fixed_pc in tqdm(dataloader, desc=f'Training for epoch: {epoch+1}/{max_epochs}', leave=False):
        moving_pc = moving_pc.to(device) # Shape: (B, N, 6)
        fixed_pc = fixed_pc.to(device)   # Shape: (B, N, 6)

        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            # Model predicts the flow (deformation)
            predicted_flow = model(moving_pc, fixed_pc) # Shape: (B, N, 3)
            
            # Warp the moving point cloud by adding the flow to its coordinates
            moving_xyz = moving_pc[..., :3]
            fixed_xyz = fixed_pc[..., :3]
            warped_xyz = moving_xyz + predicted_flow

            # Loss is calculated between the warped moving cloud and the fixed cloud
            loss = composite_loss_pointcloud(warped_xyz, fixed_xyz, predicted_flow, moving_xyz, smooth_weight=smooth_weight)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt', type=str, default='/tmp/oasis_data/train.txt')
    parser.add_argument('--template_path', type=str, default='/tmp/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2) # PointNet models can often take larger batches
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points to sample per cloud')
    parser.add_argument('--save_model_path', type=str, default='/tmp/oasis_data/weights_pointnet/')
    args = parser.parse_args()

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset")
    train_dataset = SegDataset(args.train_txt, args.template_path, num_points=args.num_points)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("Dataset loaded.")

    scaler = GradScaler()
    
    # Use the new PointNet++ model
    model = PointNet2SceneFlow(feature_channels=3).to(device) # Features are (r, theta, phi)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=1e-6
    )

    for epoch in range(args.epochs):
        avg_loss = train(
            model, train_loader, scaler, scheduler, optimizer, device, epoch, args.epochs
        )

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")
        wandb.log({'epoch': epoch + 1, 'loss': avg_loss})

        if (epoch + 1) % 20 == 0:
            model_path = f'{args.save_model_path}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    final_model_path = f'{args.save_model_path}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()