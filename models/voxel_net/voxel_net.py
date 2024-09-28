import torch
import torch.nn as nn
import torchvision.models as models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class voxelNet(nn.Module):
    def __init__(self, cfg):
        super(voxelNet, self).__init__()
        self.num_views = cfg.CONST.N_VIEWS_RENDERING
        self.embed_dim = cfg.NETWORK.EMBED_DIM
        self.combined_dim = cfg.NETWORK.COMBINED_DIM
        self.output_shape = cfg.NETWORK.OUTPUT_SHAPE
        self.num_heads = cfg.NETWORK.ATTENTION_HEADS
        
        # Feature extractor (without final classification layers)
        regnet = models.regnet_y_16gf(weights=None)
        self.feature_extractor = nn.Sequential(*list(regnet.children())[:-1])
        
        # Convert extracted features to embeddings
        self.feature_map_to_embedding = nn.Conv2d(regnet.fc.in_features, self.embed_dim, kernel_size=1)

        # Multi-head attention for view combination
        self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, batch_first=True)

        # Fully connected layers to combine and decode features
        self.fc_decoder = nn.Linear(self.embed_dim * self.num_views, self.output_shape[0] * self.output_shape[1] * self.output_shape[2])

    def forward(self, x):
        batch_size, num_views, _, _, _ = x.shape

        view_features = []
        for v in range(num_views):
            view = x[:, v]
            feature_map = self.feature_extractor(view)
            feature_map = self.feature_map_to_embedding(feature_map)
            view_features.append(feature_map.view(batch_size, -1))

        # Stack features from different views
        view_features = torch.stack(view_features, dim=1)

        # Apply multi-head attention
        attn_output, attn_weights = self.attention(view_features, view_features, view_features)
        # Log the shapes of attn_output and attn_weights
        # logging.info(f"network attn_output shape: {attn_output.shape}")
        # logging.info(f"network attn_weights shape: {attn_weights.shape}")
        attn_output = attn_output.reshape(batch_size, -1)

        # Decode to 3D output
        output_3D = self.fc_decoder(attn_output)
        output_3D = output_3D.view(-1, *self.output_shape)
        
        return output_3D, attn_weights