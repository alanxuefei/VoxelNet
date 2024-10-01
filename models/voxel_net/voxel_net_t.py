import torch
import torch.nn as nn
import torchvision.models as models
import logging

# Configure logging
logging.basicConfig(level=logging.debug, format='%(asctime)s - %(levelname)s - %(message)s')

class voxelNet(nn.Module):
    def __init__(self, cfg):
        super(voxelNet, self).__init__()
        self.num_views = cfg.CONST.N_VIEWS_RENDERING
        self.embed_dim = cfg.NETWORK.EMBED_DIM
        self.output_shape = cfg.NETWORK.OUTPUT_SHAPE
        self.num_heads = cfg.NETWORK.ATTENTION_HEADS
        
        # Feature extractor (RegNet without the final classification layer)
        regnet = models.regnet_y_16gf(weights=None)
        self.feature_extractor = nn.Sequential(*list(regnet.children())[:-1])
        
        # Project extracted features to embeddings
        self.feature_map_to_embedding = nn.Conv2d(regnet.fc.in_features, self.embed_dim, kernel_size=1)

        # First multi-head attention for combining view features
        self.attention1 = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, batch_first=True)
        
        # Second multi-head attention for processing across views
        self.attention2 = nn.MultiheadAttention(self.num_views, num_heads=1, batch_first=True)

        # Fully connected layer to decode features into the 3D voxel grid
        self.fc_decoder = nn.Linear(self.embed_dim * self.num_views, 
                                    self.output_shape[0] * self.output_shape[1] * self.output_shape[2])

    def forward(self, x):
        batch_size, num_views, _, _, _ = x.shape
        logging.debug(f"Input shape: {x.shape}")

        view_features = []
        
        # Process each view separately
        for v in range(num_views):
            view = x[:, v]  # Extract view from batch
            feature_map = self.feature_extractor(view)  # Apply feature extraction
            logging.debug(f"After feature extraction - View {v}: {feature_map.shape}")

            feature_map = self.feature_map_to_embedding(feature_map)  # Project features to embedding space
            logging.debug(f"After projection to embedding - View {v}: {feature_map.shape}")

            flattened_feature = feature_map.view(batch_size, -1)  # Flatten and store
            view_features.append(flattened_feature)
            logging.debug(f"After flattening - View {v}: {flattened_feature.shape}")

        # Stack all view features into a tensor of shape [batch_size, num_views, embedding_dim]
        view_features = torch.stack(view_features, dim=1)
        logging.debug(f"Stacked view features shape: {view_features.shape}")

        # Apply first attention (across embedding dimensions)
        attn_output, attn_weights = self.attention1(view_features, view_features, view_features)
        logging.debug(f"First attention output shape: {attn_output.shape}")
        logging.debug(f"First attention weights shape: {attn_weights.shape}")

        # Transpose the output to switch view and embedding dimensions
        attn_output = attn_output.transpose(1, 2)  # Shape becomes [batch_size, embedding_dim, num_views]
        logging.debug(f"After transposing: {attn_output.shape}")

        # Apply second attention (across views)
        attn_output, attn_weights = self.attention2(attn_output, attn_output, attn_output)
        logging.debug(f"Second attention output shape: {attn_output.shape}")
        logging.debug(f"Second attention weights shape: {attn_weights.shape}")

        # Flatten the output for the fully connected layer
        attn_output = attn_output.reshape(batch_size, -1)
        logging.debug(f"Flattened attention output shape: {attn_output.shape}")
        
        # Decode to a 3D voxel grid
        output_3D = self.fc_decoder(attn_output)
        output_3D = output_3D.view(-1, *self.output_shape)  # Reshape to the target 3D shape
        logging.debug(f"Final 3D output shape: {output_3D.shape}")
        
        return output_3D, attn_weights
