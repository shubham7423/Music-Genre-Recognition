"""
Testcases for the system
"""

import mgr.train.transformer.cnn_transformer_v3 as cnn_transformer_v3
import mgr.train.transformer.cnn_patch_transformer as cnn_patch_transformer
import mgr.train.lstm.cnn_lstm as cnn_lstm
import mgr.train.cnn. cnn as cnn
import mgr.train.cnn.resnet as resnet
import mgr 
import mgr.train
import torch
import os

def test_model_class():
    """Test the model class
    """

    assert 'torch' in globals()

    device = mgr.configuration.load_configurations()['device']
    random_inp = torch.rand(1, 1, 128, 94).to(device)
    
    model = cnn_lstm.getModel()
    opt = model(random_inp)
    assert isinstance(opt, torch.Tensor)
    assert opt.size(-1) == 8
    assert isinstance(model, torch.nn.Module)
    
    model = cnn_transformer_v3.getModel()
    opt = model(random_inp)
    assert isinstance(opt, torch.Tensor)
    assert opt.size(-1) == 8
    assert isinstance(model, torch.nn.Module)
    
    model = cnn_patch_transformer.getModel()
    opt = model(random_inp)
    assert isinstance(opt, torch.Tensor)    
    assert opt.size(-1) == 8
    assert isinstance(model, torch.nn.Module)

    train_loader, val_loader = mgr.train.data.load_data(mgr.configuration.load.load_configurations()['cnn']['train']['features_path'], mgr.configuration.load.load_configurations()['cnn']['train']['labels_path'])
    assert len(train_loader.shape) == 4
    assert len(val_loader.shape) == 4
    assert train_loader.batch_size == mgr.configuration.load.load_configurations()['cnn']['train']['train_BS']
    assert val_loader.batch_size == mgr.configuration.load.load_configurations()['cnn']['train']['valid_BS']
    assert train_loader.shape[1:] == val_loader.shape[1:]
    assert train_loader.shape[1:] == (1, 128, 94)
    assert train_loader.shape[0] == mgr.configuration.load.load_configurations()['cnn']['train']['train_BS']
    assert val_loader.shape[0] == mgr.configuration.load.load_configurations()['cnn']['train']['valid_BS']

    assert os.path.exists(mgr.configuration.load.load_configurations()['cnn']['train']['features_path'])
    assert os.path.exists(mgr.configuration.load.load_configurations()['cnn']['train']['labels_path'])
    
if __name__ == "__main__":
    test_model_class()
    print("Passed all tests")
       