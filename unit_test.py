"""
Testcases for the system
"""

import mgr.train.transformer.cnn_transformer_v3 as cnn_transformer_v3
import mgr.train.transformer.cnn_patch_transformer as cnn_patch_transformer
import mgr.train.lstm.cnn_lstm as cnn_lstm
import mgr 

import torch

def test_model_class():
    device = mgr.configuration.load_configurations()['device']
    random_inp = torch.rand(1, 1, 128, 94).to(device)
    
    model = cnn_lstm.getModel()
    opt = model(random_inp)
    assert opt.size(-1) == 8
    
    model = cnn_transformer_v3.getModel()
    opt = model(random_inp)
    assert opt.size(-1) == 8
    
    model = cnn_patch_transformer.getModel()
    opt = model(random_inp)
    assert opt.size(-1) == 8
    
if __name__ == "__main__":
    test_model_class()
    print("Passed all tests")
       