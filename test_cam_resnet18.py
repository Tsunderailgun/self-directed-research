import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import cv2
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image, deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 4
num_classes = 2
batch_size = 25   #100 -> 88% #50 -> 90% #25 -> 93% #20 -> 88.9
learning_rate = 0.001


test_transforms = transforms.Compose([transforms.Resize(size=(100, 100)),                                  
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.32306752, 0.32306752, 0.32306752],
                                                            std=[0.07111402, 0.07111402, 0.07111402])]) 
# Convolutional neural network (two convolutional layers)

#img = './img_f/PL27617-2_f_1.png'
img_n = ['./img_n/PL27614-1_n_1.png', './img_n/PL27614-1_n_2.png', './img_n/PL27614-2_n_3.png',
       './img_n/PL27616-1_n_1.png', './img_n/PL27616-2_n_3.png', './img_n/PL27619-1_n_3.png',
       './img_n/PL27619-2_n_1.png', './img_n/PL27621-1_n_1.png', './img_n/PL27621-2_n_3.png']

img_f = ['./img_f/PL27614-2_f_2.png', './img_f/PL27618-2_f_3.png', './img_f/PL27619-1_f_2.png',
         './img_f/PL27619-2_f_1.png', './img_f/PL27619-2_f_2.png', './img_f/PL27625-1_f_3.png',
         './img_f/PL27627-2_f_3.png', './img_f/PL27628-2_f_1.png', './img_f/PL27630-2_f_2.png']

model_num = 54
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

model.load_state_dict(torch.load('./ckpt_r18/' + 'model' + str(model_num) + '.ckpt'))
model.eval()

target_layers = [model.layer4[-1]]

# 2번 번갈아 수행한다. 한번은 img_f, 한번은 img_n
i = 1
for img in img_n:
    orig_pil_image = Image.open(img)
    orig_pil_image = orig_pil_image.convert('RGB')
    
    input_tensor = test_transforms(orig_pil_image).unsqueeze(0)
    
    rgb_img = cv2.imread(img, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (100,100))
    rgb_img = np.float32(rgb_img) / 255
    
    input_tensor1 = preprocess_image(rgb_img,
                                    mean=[0.32306752, 0.32306752, 0.32306752],
                                        std=[0.07111402, 0.07111402, 0.07111402])


    input_tensor = input_tensor.to(device)
    input_tensor1 = input_tensor1.to(device)

    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)

    print('predicted = ', predicted)
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    methods = {"gradcam": GradCAM}
    cam_algorithm = methods['gradcam']
    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=device) as cam:

        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor1,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite("n_" + str(i) + '.jpg', cam_image)
    cv2.imwrite("n_gb_" + str(i) + '.jpg', gb)
    cv2.imwrite("n_gb_cam" + str(i) + '.jpg', cam_gb)
    #cv2.imwrite("f_" + str(i) + '.jpg', cam_image)
    #cv2.imwrite("f_gb_" + str(i) + '.jpg', gb)
    #cv2.imwrite("f_gb_cam" + str(i) + '.jpg', cam_gb)
    i=i+1