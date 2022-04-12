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
    
    model = resnet.getModel()
    opt = model(random_inp)
    assert isinstance(opt, torch.Tensor)
    assert opt.size(-1) == 8
    assert isinstance(model, torch.nn.Module)
    
    model = cnn.getModel()
    opt = model(random_inp)
    assert isinstance(opt, torch.Tensor)
    assert opt.size(-1) == 8
    assert isinstance(model, torch.nn.Module)
    
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

    #check if the features and labels file exists
        #if you are not using all the models, comment out the model that you do not intend to use
    assert os.path.exists('mgr/'+mgr.configuration.load.load_configurations()['cnn']['train']['features_path'])
    assert os.path.exists('mgr/'+mgr.configuration.load.load_configurations()['cnn']['train']['labels_path'])
    assert os.path.exists('mgr/'+mgr.configuration.load.load_configurations()['lstm']['train']['features_path'])
    assert os.path.exists('mgr/'+mgr.configuration.load.load_configurations()['lstm']['train']['labels_path'])
    assert os.path.exists('mgr/'+mgr.configuration.load.load_configurations()['transformer']['train']['features_path'])
    assert os.path.exists('mgr/'+mgr.configuration.load.load_configurations()['transformer']['train']['labels_path'])

    #if file exists, check if they are loaded properly, as the file is common, we can use any model's path
    features, labels = mgr.train.data.load_data('mgr/'+mgr.configuration.load.load_configurations()['cnn']['train']['features_path'], 'mgr/' + mgr.configuration.load.load_configurations()['cnn']['train']['labels_path']);
    assert len(features) != 0
    assert len(labels) != 0
    assert len(features) == len(labels)
    assert list(features.shape[1:]) == [1, 128, 94]

    #check if the preprocessing part directory exists
    assert os.path.exists('mgr/' + mgr.configuration.load.load_configurations()['preprocessing']['train'])
    assert os.path.exists('mgr/' + mgr.configuration.load.load_configurations()['preprocessing']['test'])
    
    #further validations of train_loader and val_loader variables using features and labels dataset
    train_loader, val_loader = mgr.train.data.get_data(features, labels);
    assert len(train_loader) != 0
    assert len(val_loader) != 0
        
    #check the batch size is properly maintained or not
    assert train_loader.batch_size == mgr.configuration.load.load_configurations()['cnn']['train']['train_BS']
    assert val_loader.batch_size == mgr.configuration.load.load_configurations()['cnn']['train']['valid_BS']
    assert train_loader.batch_size == mgr.configuration.load.load_configurations()['lstm']['train']['train_BS']
    assert val_loader.batch_size == mgr.configuration.load.load_configurations()['lstm']['train']['valid_BS']
    assert train_loader.batch_size == mgr.configuration.load.load_configurations()['transformer']['train']['train_BS']
    assert val_loader.batch_size == mgr.configuration.load.load_configurations()['transformer']['train']['valid_BS']
    
    #check if the folder exists where we save the models
    assert os.path.exists('mgr/' + mgr.configuration.load.load_configurations()['cnn']['train']['save_model_at'])
    assert os.path.exists('mgr/' + mgr.configuration.load.load_configurations()['lstm']['train']['save_model_at'])
    assert os.path.exists('mgr/' + mgr.configuration.load.load_configurations()['transformer']['train']['save_model_at'])
    
    del(model)
    del(opt)
    del(train_loader)
    del(val_loader)
    del(features)
    del(labels)
    del(device)
    del(random_inp)
    
if __name__ == "__main__":
    test_model_class()
    print("Passed all tests")
       