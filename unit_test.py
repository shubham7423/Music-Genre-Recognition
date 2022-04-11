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

    train_loader, val_loader = mgr.train.get_data(mgr.configration.load_configurations()['data_path'])
    assert len(train_loader.shape) == 4
    assert len(val_loader.shape) == 4
    assert train_loader.batch_size == mgr.configration.load_configurations()['train_batch_size']
    assert val_loader.batch_size == mgr.configration.load_configurations()['valid_batch_size']
    assert train_loader.shape[1:] == val_loader.shape[1:]
    assert train_loader.shape[1:] == (1, 128, 94)
    assert train_loader.shape[0] == mgr.configration.load_configurations()['train_size']
    assert val_loader.shape[0] == mgr.configration.load_configurations()['valid_size']


    
if __name__ == "__main__":
    test_model_class()
    print("Passed all tests")
       