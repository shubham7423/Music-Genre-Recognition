from mgr.model.transformer import cnn_transformer_v3
from mgr.train.data import get_data
from mgr.train.model_utils import get_model_configs
from mgr.trainer import Trainer
from mgr.configuration import load_configurations

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def start_training():
    """Create the model
    
    Returns:
    ________
    model: torch.nn.Module
        Model to be trained
    """

    CFG = load_configurations()['transformer']['train']
    device = load_configurations()['device']

    model = cnn_transformer_v3.CNNTransformerV3(
        embed_dim=CFG['model_params']['embed_dim'],
        hidden_dim=CFG['model_params']['hidden_dim'],
        num_head=CFG['model_params']['num_heads'],
        num_layers=CFG['model_params']['num_layers'],
        num_patches=CFG['model_params']['num_patches'],
        num_classes=CFG['model_params']['num_classes'],
        dropout=CFG['model_params']['dropout'],
        device=device
    )

    transform = transforms.Compose([
        transforms.Normalize(0.5, 0.5)
    ])

    train_loader, val_loader = get_data(CFG['features'], CFG['labels'], transform)
    criterion, optimizer, scheduler = get_model_configs(model, train_loader)

    trainer = Trainer(model, device, optimizer, criterion, scheduler)

    History = trainer.fit(CFG['early_stopping'], CFG['save_model_at'], train_loader, 
                          val_loader, CFG['epochs'], train_BS=CFG['train_BS'], valid_BS=CFG['valid_BS'])

    plt.plot(History['train_loss'], label="train")
    plt.plot(History['val_loss'], label="val")
    plt.title("Loss")
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

    plt.plot(History['train_acc'], label="train")
    plt.plot(History['val_acc'], label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel('epochs')
    plt.show()

    return model, History