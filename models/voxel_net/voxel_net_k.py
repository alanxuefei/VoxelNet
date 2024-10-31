import torch
import torch.nn as nn
import torchvision.models as models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoxelNet(nn.Module):
    def __init__(self, cfg):
        super(VoxelNet, self).__init__()
        self.embed_dim = cfg.NETWORK.EMBED_DIM
        self.output_shape = cfg.NETWORK.OUTPUT_SHAPE
        self.num_heads = cfg.NETWORK.ATTENTION_HEADS
        
        # Feature extractor (RegNet without the final classification layer)
        regnet = models.regnet_y_16gf(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(regnet.children())[:-1])

        # Convert extracted features to embeddings
        self.feature_map_to_embedding = nn.Conv2d(regnet.fc.in_features, self.embed_dim, kernel_size=1)

        # Multi-head attention for view combination
        self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, batch_first=True)
        
        # Fully connected layer to decode the combined features
        self.fc_decoder = nn.Linear(self.embed_dim, self.output_shape[0] * self.output_shape[1] * self.output_shape[2])

        # Output normalization layer
        self.output_norm = nn.LayerNorm(self.output_shape)  # Normalize across D, H, W dimensions

    def forward(self, x):
        batch_size, num_views, _, _, _ = x.shape
        logging.debug(f"Input shape: {x.shape}")

        # Extract features for each view
        view_features = []
        for v in range(num_views):
            view = x[:, v]
            logging.debug(f"Processing view {v + 1}/{num_views}, view shape: {view.shape}")
            feature_map = self.feature_extractor(view)
            logging.debug(f"Feature map shape after feature extractor: {feature_map.shape}")
            feature_map = self.feature_map_to_embedding(feature_map)
            logging.debug(f"Feature map shape after embedding: {feature_map.shape}")
            feature_map = feature_map.view(batch_size, -1)
            view_features.append(feature_map)

        # Stack features from different views
        view_features = torch.stack(view_features, dim=1)  # Shape: (batch_size, num_views, embed_dim)
        logging.debug(f"Stacked view features shape before attention: {view_features.shape}")

        # Apply multi-head attention across views
        view_features_attn, attn_weights = self.attention(view_features, view_features, view_features)
        logging.debug(f"Attention output shape: {view_features_attn.shape}")
        logging.debug(f"Attention weights shape: {attn_weights.shape}")

        # Max pooling across views for both the original and attention-transformed features
        max_view_features, _ = torch.max(view_features, dim=1)       # Max across original view features
        max_view_features_attn, _ = torch.max(view_features_attn, dim=1)  # Max across attention view features
        logging.debug(f"Max pooled original features shape: {max_view_features.shape}")
        logging.debug(f"Max pooled attention features shape: {max_view_features_attn.shape}")

        # Take the element-wise maximum of the max-pooled original and attention-transformed features
        combined_features = torch.max(max_view_features, max_view_features_attn)
        logging.debug(f"Combined features shape after max: {combined_features.shape}")

        # Decode to 3D output using a single fully connected layer
        output_3D = self.fc_decoder(combined_features)
        output_3D = output_3D.view(-1, *self.output_shape)  # Reshape to the output shape
        logging.debug(f"Output 3D shape before normalization: {output_3D.shape}")

        # Apply normalization to the output
        # output_3D = self.output_norm(output_3D)
        logging.debug(f"Output 3D shape after normalization: {output_3D.shape}")
        
        return output_3D
