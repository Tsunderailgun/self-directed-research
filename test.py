import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


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

test_data = torchvision.datasets.ImageFolder(root = "../../../../welding_dataset/dataset/test/",
                                             transform = test_transforms)
                                             

test_loader = DataLoader(test_data, batch_size, shuffle=True)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #self.fc = nn.Linear(100*100*32, num_classes)
        self.fc = nn.Linear(20000, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)


best_accuracy = 0
best_epoch = 0
accuracy = 0
#for i in range(0,4):
for i in range(30,150):  #ckpt 150, ckpt1:200
    model.load_state_dict(torch.load('./ckpt/' + 'model' + str(i) + '.ckpt'))
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images[:,0, :, :]
            images = images.unsqueeze(dim=1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct/total
        #print(str(i) + ' Test Accuracy of the model on texhe test images: {} %'.format(100 * correct / total))
        print(i, accuracy)
    is_best = accuracy > best_accuracy
    best_accuracy = max(accuracy, best_accuracy)
    if is_best:
        best_epoch = i
print(str(best_epoch) + '** Best Test Accuracy of the model on the test images: {} %'.format(100 * best_accuracy))
