import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        def upsample_block(in_channels, out_channels):
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # Encoder
        self.enc1 = conv_block(in_channels, 32)  # 10 channels: template + fixed given together as input 5 channels each
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        
        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up4 = upsample_block(512, 256)
        self.dec4 = conv_block(512, 256)
        self.up3 = upsample_block(256, 128)
        self.dec3 = conv_block(256, 128)
        self.up2 = upsample_block(128, 64)
        self.dec2 = conv_block(128, 64)
        self.up1 = upsample_block(64, 32)
        self.dec1 = conv_block(64, 32)

        
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)  # 3 channels for deformation field (x, y, z)

    def forward(self, x):
        # Encoder part
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        #Decoder part
        up4 = self.up4(b)
        d4 = self.dec4(torch.cat((up4, e4), dim=1))
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))

     
        deformation_field = self.out_conv(d1)
        return deformation_field
    
    
# This model takes a sparse point cloud and outputs a dense 3D deformation field.
class PointNetDenseFlow(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, global_feature_size=1024, output_size=(64, 64, 64)):
        super(PointNetDenseFlow, self).__init__()
        
        # --- PointNet Encoder Part ---
        # Learns features from each point and aggregates them into a global vector.
        self.encoder_mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.encoder_mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, global_feature_size, 1), nn.BatchNorm1d(global_feature_size), nn.ReLU()
        )
        
        # --- CNN Decoder Part ---
        # Upsamples the global feature vector into a dense 3D flow field.
        # The decoder starts from a 4x4x4 volume.
        self.decoder_start_size = 4
        self.decoder_fc = nn.Linear(global_feature_size, 512 * self.decoder_start_size**3)

        def upsample_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        self.decoder_cnn = nn.Sequential(
            upsample_block(512, 256), # 4x4x4 -> 8x8x8
            upsample_block(256, 128), # 8x8x8 -> 16x16x16
            upsample_block(128, 64),  # 16x16x16 -> 32x32x32
            upsample_block(64, 32),   # 32x32x32 -> 64x64x64
            # Final convolution to produce the 3-channel flow field.
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        )
        
        # Initialize final layer weights to be small to encourage starting near identity transform
        self.decoder_cnn[-1].weight.data.normal_(0, 1e-5)
        self.decoder_cnn[-1].bias.data.zero_()


    def forward(self, x_points):
        """
        Args:
            x_points (torch.Tensor): Input point cloud, shape (B, num_features, num_points).
        """
        # --- Encoder ---
        local_features = self.encoder_mlp1(x_points)
        global_features_intermediate = self.encoder_mlp2(local_features)
        
        # Symmetric pooling (max) to get the global feature vector
        global_feature, _ = torch.max(global_features_intermediate, dim=2) # Shape: (B, global_feature_size)
        
        # --- Decoder ---
        # Start the upsampling process
        x = self.decoder_fc(global_feature)
        x = x.view(x.size(0), 512, self.decoder_start_size, self.decoder_start_size, self.decoder_start_size)
        
        # Generate the dense flow field
        dense_flow = self.decoder_cnn(x)
        
        return dense_flow