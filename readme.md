The repo contain generic code files for computer vision problems. 



#### File Structure



- `main.py`
  - `get_data_loader()` -> Returns train test data loader for the given dataset and batch size.
  - `get_model()` -> Returns requested model from model directory
  - `set_model_config()` -> It sets model config like optimizer, scheduler, type of device, learning rate.
  - `train()` -> Contains training logic of the model
  - `test()` -> Contains testing logic of the model
  - `training_loop()` -> Contains logic of training and testing the model for the given number of epochs.
- utils
  - `Albumentationtransform.py` -> Contains logic for albumentation transformation.
  - `generic_func.py` -> Contains functions for downloading train and test set and plotting curves after model training.
  - `gradcam.py` -> Contains logic code for gradcam
  - `utils_grad_cam.py` -> Contains utils functions for gradcam
  - `missclassified_images.py` -> Contains function to plot miss classified images by inference on test loader. 
- models
  - `resnet.py` : Contains model code for resnet18 and resnet34