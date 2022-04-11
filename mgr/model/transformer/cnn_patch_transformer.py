import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


def patch(features, kernel_size=(9, 14), stride=(6, 10)):
    """Patch the features

    Arguments:
    __________
    features: torch.Tensor
        Features of the data
    kernel_size: tuple
        Size of the kernel
    stride: tuple
        Stride of the kernel

    Returns:
    ________
    torch.Tensor
        Patched features
    """
    features = features.unfold(
        2, kernel_size[0], stride[0]).unfold(
        3, kernel_size[1], stride[1])
    features = features.permute(0, 2, 3, 1, 4, 5)
    features = features.flatten(1, 2)
    features = features.flatten(2, 4)
    return features


class PatchTransformer(nn.Module):
    """Patch Transformer

    Arguments:
    __________
    embed_dim: int
        Dimension of embedding
    hidden_dim: int
        Dimension of hidden layer
    num_channels: int
        Number of channels
    num_head: int
        Number of heads
    num_layers: int
        Number of layers
    num_classes: int
        Number of classes
    dropout: float
        Dropout rate
    h_patch: int
        Height of the patch
    w_patch: int
        Width of the patch
    h_stride: int
        Height of the stride
    w_stride: int
        Width of the stride
    num_patches: int
        Number of patches
    device: str
        Device to use
    """

    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_channels,
            num_head,
            num_layers,
            num_classes,
            dropout=0.0,
            h_patch=9,
            w_patch=14,
            h_stride=2,
            w_stride=2,
            num_patches=33,
            device="cpu"):
        super().__init__()
        self.device = device
        self.patch_size = (h_patch, w_patch)
        self.stride = (h_stride, w_stride)

        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=(
                7, 7), stride=(
                2, 2), padding=(
                3, 3), bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-5])

        self.input_layer = nn.Linear(512, embed_dim)
        self.transformer = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_head, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        """Forward pass

        Arguments:
        __________
        x: torch.Tensor
            Input tensor

        Returns:
        ________
        out: torch.Tensor
            Output tensor
        """

        x = self.resnet(x)

        x = x.flatten(2, 3)

        x = x.unsqueeze(1)
        x = patch(x, kernel_size=(self.patch_size), stride=(self.stride))

        B, T, _ = x.shape

        x = self.input_layer(x)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        out = self.mlp_head(cls)
        return out


class AttentionBlock(nn.Module):
    """Attention Block

    Arguments:
    __________
    embed_dim: int
        Dimension of embedding
    hidden_dim: int
        Dimension of hidden layer
    num_heads: int
        Number of heads
    dropout: float
        Dropout rate
    """

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass

        Arguments:
        __________
        x: torch.Tensor
            Input tensor

        Returns:
        ________
        out: torch.Tensor
            Output tensor
        """
        input_x = self.layer_norm_1(x)
        x = x + self.attn(input_x, input_x, input_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
