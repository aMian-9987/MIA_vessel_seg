import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .KRFormer import KRtransformer, KRTransformer_Test
# Assume the KRtransformer class definition is already provided and included
#    self.encoder1 = UnetrBasicBlock(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#             res_block=True,
#         )
# Helper function to ensure tuple format
def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        # Define the ResBlock layers
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return F.relu(out)    
class PatchEmbedding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(num_patches, embedding_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=2)  # Flatten the spatial dimensions
        x = self.projection(x)
        return x
    
class GrainedFusion(nn.Module):
    def __init__(self, num_patches, embedding_dim, num_channels):
        super().__init__()
        self.patch_embedding = PatchEmbedding(num_patches, embedding_dim)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_channels),
            nn.ReLU(inplace=True)
        )
        self.final_linear = nn.Linear(num_channels, num_channels)
        self.final_norm = nn.LayerNorm(num_channels)

    def forward(self, x_c, x_t):
        # Assuming x_c is 4D and x_t is 3D
        x_c = self.patch_embedding(x_c)  # Convert 4D to 3D
        x_t = x_c + x_t  # Element-wise summation
        x1 = self.global_pooling(x_t).squeeze(-1)  # Global max pooling
        x2 = self.linear1(x1)
        x3 = self.norm1(x2)
        x4 = F.softmax(x3, dim=-1)  # Softmax
        
        # Matrix multiplication (bmm)
        x_t = x_t.transpose(1, 2)  # Prepare for bmm
        x = torch.bmm(x4, x_t)
        x = self.linear2(x)
        # Reshape and concatenate with original input
        x = x.view(x.size(0), -1, x_c.size(2), x_c.size(3), x_c.size(4))
        x = torch.cat((self.conv1(x), x), dim=1)
        x = self.conv2(x)
        
        # Final linear layer and normalization
        x = self.final_linear(x)
        x = self.final_norm(x)
        
        return x

# Fusion-module that uses ResBlock, KRtransformer and GrainedFusion
class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, layer_depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super(FusionModule, self).__init__()
        self.resblock = ResBlock(in_channels)
        self.downsample_conv = nn.Conv3d(in_channels, in_channels, kernel_size=7, stride=2, padding=3)
        self.downsample_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.krtransformer = KRtransformer(in_channels, out_channels, image_size, layer_depth, dmodel, mlp_dim, patch_size, heads, dim_head, dropout, emb_dropout)
        
        image_height, image_width, image_depth = pair(image_size)
        self.grained_fusion = GrainedFusion(
            num_patches=(image_height // patch_size) * (image_width // patch_size) * (image_depth // patch_size),
            embedding_dim=dmodel,
            in_channels=out_channels,
            transformer_output_dim=dmodel
        )

    def forward(self, x):
        # Apply ResBlock
        x_r = self.resblock(x)

        # Downsampling
        x_r = self.downsample_conv(x_r)
        x_r = self.downsample_pool(x_r)
        x_t, _, _ = self.krtransformer(x)
        x_rt = self.grained_fusion(x_r, x_t)

        return x_rt
    
class DoubleFusion(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, image_size, layer_depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super(DoubleFusion, self).__init__()
        
        # First FusionModule
        self.fusion_module_1 = FusionModule(in_channels, mid_channels, image_size, layer_depth, dmodel, mlp_dim, patch_size, heads, dim_head, dropout, emb_dropout)
        
        # Assuming the image size is halved by the downsampling in FusionModule, adjust for the second FusionModule
        reduced_image_size = (image_size[0] // 4, image_size[1] // 4, image_size[2] // 4) # Update this based on actual downsampling in FusionModule
        self.fusion_module_2 = FusionModule(mid_channels, out_channels, reduced_image_size, layer_depth, dmodel, mlp_dim, patch_size, heads, dim_head, dropout, emb_dropout)
    
    def forward(self, x):
        # First fusion module
        x = self.fusion_module_1(x)
        
        # Second fusion module
        x = self.fusion_module_2(x)
        
        return x


# Instantiate the FusionModule
# in_channels = 1
# out_channels = 1
# image_size = (96, 96, 96) 
# fusion_module = FusionModule(in_channels, out_channels, image_size)

# Example input tensor  # 