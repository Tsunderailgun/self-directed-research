import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models
from torchsummary import summary   ########


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 200
num_classes = 2
batch_size = 25   #100 -> 88% #50 -> 90% #25 -> 93% #20 -> 88.9
learning_rate = 0.001

train_transforms = transforms.Compose([transforms.Resize(size=(100, 100)),
                                       transforms.RandomHorizontalFlip(p=0.5),                                       
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.32306752, 0.32306752, 0.32306752],
                                                            std=[0.07111402, 0.07111402, 0.07111402])]) 

test_transforms = transforms.Compose([transforms.Resize(size=(100, 100)),                                  
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.32306752, 0.32306752, 0.32306752],
                                                            std=[0.07111402, 0.07111402, 0.07111402])]) 

train_data = torchvision.datasets.ImageFolder(root = "../../../welding_dataset/dataset/train",
                                             transform = train_transforms)
test_data = torchvision.datasets.ImageFolder(root = "../../../welding_dataset/dataset/test/",
                                             transform = test_transforms)
                                             
for num, value in enumerate(train_data):
    image, label = value
    #print(num, image, label)
    img = image[0].numpy()
    #plt.imshow(np.transpose(image, (1, 2, 0)))
    #plt.show()
    break                                             

#train_loader = DataLoader(train_data, batch_size, shuffle=True)
#test_loader = DataLoader(test_data, batch_size, shuffle=True)

#train validation 을 9:1 비율로 나누는데 랜덤으로 셔플하여 9대1로 나눈후에 
# train, validation이 나누어지면 dataloader에서 샘플을 가져올때 다시 랜덤으로 추출하여 가져오기 위해 
# SubsetRandomSampler를 사용함

validation_split = 0.1  # 비율을 얼마로 줄것인지?
shuffle_dataset = True
random_seed= 42

t_size = len(train_data)
indices = list(range(t_size))
split = int(np.floor(validation_split * t_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

#최종 train, validation, test가 모두 정의되었으므로 loader를 호출한다.
train_loader = DataLoader(train_data, batch_size, sampler=train_sampler)
validation_loader = DataLoader(train_data, batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_data, batch_size, shuffle=True)

# Resnet18 Convolutional neural network 
resnet18_pretrained_model = models.resnet18(pretrained=True)
#print(resnet18_pretrained_model)
num_ftrs = resnet18_pretrained_model.fc.in_features
resnet18_pretrained_model.fc = nn.Linear(num_ftrs, num_classes)
resnet18_pretrained_model.to(device)

summary(resnet18_pretrained_model, input_size=(3, 224, 224), device=device.type)
summary(resnet18_pretrained_model, input_size=(3, 100, 100), device=device.type)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
optimizer = torch.optim.Adam(resnet18_pretrained_model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)

#best_accuracy = 0
for epoch in range(num_epochs):
    resnet18_pretrained_model.train()
    for i, (images, labels) in enumerate(train_loader):
     #   images = images[:,0, :, :]
     #   images = images.unsqueeze(dim=1)
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = resnet18_pretrained_model(images)
        print(resnet18_pretrained_model)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % batch_size == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    resnet18_pretrained_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_loader:
        #    images = images[:,0, :, :]
        #    images = images.unsqueeze(dim=1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet18_pretrained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #accuracy = correct/total
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    #is_best = accuracy > best_accuracy
    #best_accuracy = max(accuracy, best_accuracy)
    #if is_best:
    #if epoch >= 30:
    torch.save(resnet18_pretrained_model.state_dict(), './ckpt_r18/' + 'model'+ str(epoch) + '.ckpt')
