import torch
import torch.nn as nn
import torchvision.models as models
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def positionalencoding1d(d_model, length):
    """
    Generates a 1D positional encoding matrix.
    :param d_model: dimension of the model (embedding dimension)
    :param length: length of positions (number of views)
    :return: A (length, d_model) position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

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

        # Multi-head attention for view combination
        self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, batch_first=True)
        self.intra_attention = nn.MultiheadAttention(int(self.embed_dim / 96), num_heads=self.num_heads, batch_first=True)
        # Fully connected layer to decode the combined features
        self.fc_decoder = nn.Linear(self.embed_dim, self.output_shape[0] * self.output_shape[1] * self.output_shape[2])

    def forward(self, x):
        batch_size, num_views, _, _, _ = x.shape
        logging.debug(f"Input shape: {x.shape}")

        # Extract features for each view and apply positional encoding
        view_features = []
        for v in range(num_views):
            view = x[:, v]
            logging.debug(f"Processing view {v + 1}/{num_views}, view shape: {view.shape}")
            feature_map = self.feature_extractor(view)
            logging.debug(f"Feature map shape after feature extractor: {feature_map.shape}")
            feature_map = self.feature_map_to_embedding(feature_map)
            logging.debug(f"Feature map shape after embedding: {feature_map.shape}")

            # Flatten and split the feature map into smaller chunks (patches)
            feature_map = feature_map.view(batch_size, -1)  # Shape: (batch_size, embed_dim)
            num_patches = self.num_patches
            split_dim = self.embed_dim // num_patches
            split_features = torch.split(feature_map, split_dim, dim=1)  # Split embeddings into smaller patches

            # Generate positional encoding for the patches within the view
            patch_positional_encoding = positionalencoding1d(split_dim, len(split_features)).to(x.device)

            # Apply positional encoding to each split patch
            encoded_splits = []
            for i, split in enumerate(split_features):
                encoded_split = split + patch_positional_encoding[i]
                encoded_splits.append(encoded_split)

            # Apply intra-view attention on the encoded splits
            encoded_splits_stack = torch.stack(encoded_splits, dim=1)  # Shape: (batch_size, num_patches, split_dim)
            intra_view_attn, _ = self.intra_attention(encoded_splits_stack, encoded_splits_stack, encoded_splits_stack)
            encoded_splits = [intra_view_attn[:, i, :] for i in range(num_patches)]

            # Concatenate the encoded splits back to original embedding dimension
            restored_feature_map = torch.cat(encoded_splits, dim=1)
            view_features.append(restored_feature_map)

        # Stack features from different views
        view_features = torch.stack(view_features, dim=1)  # Shape: (batch_size, num_views, embed_dim)
        logging.debug(f"Stacked view features shape before attention: {view_features.shape}")

        # Apply multi-head attention across views
        view_features_attn, attn_weights = self.attention(view_features, view_features, view_features)
        logging.debug(f"Attention output shape: {view_features_attn.shape}")
        logging.debug(f"Attention weights shape: {attn_weights.shape}")

        # Decode each view feature separately after attention and sum them up
        decoded_outputs = []
        for v in range(num_views):
            decoded_view = self.fc_decoder(view_features_attn[:, v, :])  # Decode each view
            decoded_outputs.append(decoded_view.view(-1, *self.output_shape))  # Reshape to output shape

        # Sum the decoded outputs from each view
        output_3D = torch.stack(decoded_outputs, dim=0).sum(dim=0)  # Sum across views
        logging.debug(f"Final 3D output shape after summing decoded views: {output_3D.shape}")

        return output_3D