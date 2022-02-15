## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net9(nn.Module):

    def __init__(self):
        super(Net9, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #obejctive is to bring down the image size to single unit-->
        #here given image size is 224x224px
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256,512,1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(6*6*512 , 1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,136)
        
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        x = self.drop5(self.pool(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    

class Net13(nn.Module):

    def __init__(self):
        super(Net13, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # netwerk architecture taken from https://arxiv.org/pdf/1710.00977.pdf, with an adjustment
        # in the last linear layer to detect all keypoints at once
        # plus the input is assumed to be of size 1 x 96 x 96 as in the naimishnet paper
        
        self.conv1 = nn.Conv2d(1, 32, 4, stride=1)     #  32 x 221 x 221 -- pool:  32 x 110 x 110
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)    #  64 x 108 x 108 -- pool:  64 x  54 x  54
        self.conv3 = nn.Conv2d(64, 128, 2, stride=1)   # 128 x  53 x  53 -- pool: 128 x  26 x  26 --> 86528
        self.conv4 = nn.Conv2d(128, 256, 1, stride=1)  # 256 x  26 x  26 -- pool: 256 x  13 x  13 --> 43264
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # blocks of conv - activation - pooling - dropout
        x = self.dropout1(self.pool(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool(F.elu(self.conv4(x))))
        
        # prep for linear layers -- flatten
        x = x.view(x.size(0), -1)
        
        # dense layers
        x = F.elu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class Net14(nn.Module):

    def __init__(self):
        super(Net14, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # netwerk architecture taken from https://arxiv.org/pdf/1710.00977.pdf, with an adjustment
        # in the last linear layer to detect all keypoints at once
        # plus the input is assumed to be of size 1 x 96 x 96 as in the naimishnet paper
        
        self.conv1 = nn.Conv2d(1, 32, 4, stride=1)     #  32 x 221 x 221 -- pool:  32 x 110 x 110
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)    #  64 x 108 x 108 -- pool:  64 x  54 x  54
        self.conv3 = nn.Conv2d(64, 128, 2, stride=1)   # 128 x  53 x  53 -- pool: 128 x  26 x  26 --> 86528
        self.conv4 = nn.Conv2d(128, 256, 1, stride=1)  # 256 x  26 x  26 -- pool: 256 x  13 x  13 --> 43264
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # blocks of conv - activation - pooling - dropout
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        
        # prep for linear layers -- flatten
        x = x.view(x.size(0), -1)
        
        # dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
    