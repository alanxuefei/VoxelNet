import torch
import torch.nn as nn
import torchvision.models as models
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def positionalencoding1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

class DivergenceEnhancedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DivergenceEnhancedMultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Linear layer to project concatenated features back to original dimension
        self.projection = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        
        # Concatenate original input and attention output to promote divergence
        enhanced_output = torch.cat((x, attn_output), dim=-1)  # Concatenates along the embedding dimension
        
        # Project concatenated result back to the original embedding dimension
        enhanced_output = self.projection(enhanced_output)
        
        return enhanced_output, attn_weights

class VoxelNet(nn.Module):
    def __init__(self, cfg):
        super(VoxelNet, self).__init__()
        self.embed_dim = cfg.NETWORK.EMBED_DIM
        self.output_shape = cfg.NETWORK.OUTPUT_SHAPE
        self.num_heads = cfg.NETWORK.ATTENTION_HEADS
        self.num_patches = 96

        # Feature extractor (RegNet without the final classification layer)
        regnet = models.regnet_y_16gf(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(regnet.children())[:-1])

        # Convert extracted features to embeddings
        self.feature_map_to_embedding = nn.Conv2d(regnet.fc.in_features, self.embed_dim, kernel_size=1)

        # Divergence-Enhanced Multi-head attention for view combination
        self.attention = DivergenceEnhancedMultiHeadAttention(self.embed_dim, num_heads=self.num_heads)
        self.attention1 = nn.MultiheadAttention(int(self.embed_dim / 96), num_heads=self.num_heads, batch_first=True)

        # Fully connected layer for decoding
        self.fc_decoder = nn.Linear(self.embed_dim, self.output_shape[0] * self.output_shape[1] * self.output_shape[2])

    def forward(self, x):
        batch_size, num_views, _, _, _ = x.shape
        logging.debug(f"Input shape: {x.shape}")

        # Extract features for each view and apply positional encoding
        view_features = []
        for v in range(num_views):
            view = x[:, v]
            feature_map = self.feature_extractor(view)
            feature_map = self.feature_map_to_embedding(feature_map)
            feature_map = feature_map.view(batch_size, -1)  # Flatten

            # num_patches = self.num_patches
            # split_dim = self.embed_dim // num_patches
            # split_features = torch.split(feature_map, split_dim, dim=1)

            # patch_positional_encoding = positionalencoding1d(split_dim, len(split_features)).to(x.device)
            # encoded_splits = [split + patch_positional_encoding[i] for i, split in enumerate(split_features)]
            # encoded_splits_stack = torch.stack(encoded_splits, dim=1)
            # intra_view_attn, _ = self.attention1(encoded_splits_stack, encoded_splits_stack, encoded_splits_stack)
            # encoded_splits = [intra_view_attn[:, i, :] for i in range(num_patches)]

            # restored_feature_map = torch.cat(feature_map, dim=1)
            view_features.append(feature_map)

        # Stack features from different views
        view_features = torch.stack(view_features, dim=1)  # Shape: (batch_size, num_views, embed_dim)
        logging.debug(f"Stacked view features shape before attention: {view_features.shape}")

        # Apply divergence-enhanced attention across views
        view_features_attn, attn_weights = self.attention(view_features)
        logging.debug(f"Enhanced attention output shape: {view_features_attn.shape}")
        logging.debug(f"Attention weights shape: {attn_weights.shape}")

        # Decode each view feature separately after attention and average them
        decoded_outputs = []
        for v in range(num_views):
            decoded_view = self.fc_decoder(view_features_attn[:, v, :])  # Decode each view
            decoded_outputs.append(decoded_view)  # Append decoded output for each view

        # Stack along a new dimension for max operation
        decoded_outputs_tensor = torch.stack(decoded_outputs, dim=1)  # Shape: (batch_size, num_views, output_dim)
        output_3D, _ = torch.max(decoded_outputs_tensor, dim=1)  # Max over views, resulting in shape: (batch_size, output_dim)

        output_3D = output_3D.view(batch_size, 32, 32, 32)
        decoded_outputs = [output.view(batch_size, 32, 32, 32) for output in decoded_outputs]
        decoded_outputs = torch.stack(decoded_outputs, dim=1) 
        return output_3D, decoded_outputs
