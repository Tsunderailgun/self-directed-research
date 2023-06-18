# self-directed research (2023 spring semester )
2023-1 Self-directed research 

This project is results of self-directed research.

We built a welding dataset from weld X-ray images. Using this dataset, we trained two models, the simple-CNN model and the resnet18 model, to classify the defective and normal areas in the weld X-ray images. 
To verify that the resulting models were properly trained, we used grad-CAM to examine the regions where classes are activated.

# utility
getmean.py : get mean and std from welding dataset for normalization

# Training Simple CNN 2-layer model
train.py : training code of Simple CNN 2-layer model 

test.py : testing code of Simple CNN 2-layer model 

# Training Simple CNN 2-layer model
train_resnet18.py : training code of Resnet18 model

test_resnet18.py : testing code of Resnet18 model

# grad-CAM Visualization
test_cam_conv.py : applying gradCAM to Simple CNN 2-layer model for visualiztion 

test_cam_resnet18.py : applying gradCAM to Resnet18 model for visualiztion 
