import torch
# import trainer
from model import cnn_transformer_v3
import train.data as data



def create_model(
    embed_dim=256,
    hidden_dim=512,
    num_heads=4,
    num_layers=32,
    dropout=0.3,
    num_classes=8,
    num_patches=65,
    device="cpu",
):
    """Create the model
    
    Returns:
    ________
    model: torch.nn.Module
        Model to be trained
    """
    model = cnn_transformer_v3.CNNTransformerV3(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_head=num_heads,
        num_layers=num_layers,
        num_patches=num_patches,
        num_classes=num_classes,
        dropout=dropout,
        device=device
    )
    train.g
    train_loader, val_loader = get_data(features, labels, transform)
    
    criterion, optimizer, scheduler = get_model_configs(model, train_loader)
    return model