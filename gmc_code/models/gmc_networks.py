import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import CoordEncoder

# COMMON

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class SharedProjection(nn.Module):

    def __init__(self, common_dim, latent_dim):
        super(SharedProjection, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)

    

class ImageFeaturePairProcessor(nn.Module):

    def __init__(self, common_dim):
        super(ImageFeaturePairProcessor, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(2*1024, common_dim)

    def forward(self, x):
        return self.projector(x)


class CoordProcessor(nn.Module):
    def __init__(self, common_dim):
        super(CoordProcessor, self).__init__()
        self.common_dim = common_dim

        # Process goal position [3] -> [common_dim]
        self.coord_encoder = CoordEncoder(
            fourier_sigma=10.0,
            fourier_m=16,
            coord_dim=3,
            hidden_dim=128,
            output_dim=64,
            depth=4,
            use_layernorm=True,
        )
        self.projector = nn.Linear(64, common_dim)

    def forward(self, x):
        # x shape: [batch_size, 3]
        h = self.coord_encoder(x)  # [batch_size, 64]
        return self.projector(h)  # [batch_size, common_dim]


class JointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(JointProcessor, self).__init__()
        self.common_dim = common_dim

        self.image_features_processor = nn.Linear(2*1024, common_dim)
        self.position_processor = CoordEncoder(
            fourier_sigma=10.0,
            fourier_m=16,
            coord_dim=3,
            hidden_dim=128,
            output_dim=64,
            depth=6,
            use_layernorm=True,
        )

        # Joint projection
        self.joint_projector = nn.Linear(common_dim + 64, common_dim)

    def forward(self, x):
        # x is a list: [image_features, goal_position]
        image_features, goal_position = x[0], x[1]

        # Process each modality
        h_image_features = self.image_features_processor(image_features)
        h_goal_position = self.position_processor(goal_position)

        # Concatenate all representations
        joint_repr = torch.cat([h_image_features, h_goal_position], dim=-1)

        return self.joint_projector(joint_repr)

