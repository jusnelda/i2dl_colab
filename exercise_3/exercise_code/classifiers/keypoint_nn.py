import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        # CONVOLUTIONS
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        # ACTIVATION
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        # POOLING
        self.maxpool = nn.MaxPool2d(2, stride=2)
        # DROPOUT
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout4 = nn.Dropout2d(0.4)
        self.dropout5 = nn.Dropout2d(0.5)
        self.dropout6 = nn.Dropout2d(0.6)
        # DENSE
        self.dense1 = nn.Linear(6400, 1000)
        self.dense2 = nn.Linear(1000, 1000)
        self.dense3 = nn.Linear(1000, 30)
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.conv1(x)       # 2 Convolution2d_1
        x = self.elu(x)         # 3 Activation_1
        x = self.maxpool(x)     # 4 Maxpooling2d_1
        x = self.dropout1(x)    # 5 Dropout_1
        x = self.conv2(x)       # 6 Convolution2d_2
        x = self.elu(x)         # 7 Activation_2
        x = self.maxpool(x)     # 8 Maxpooling2d_2
        x = self.dropout2(x)    # 9 Dropout_2
        x = self.conv3(x)       # 10 Convolution2d_3
        x = self.elu(x)         # 11 Activation_3
        x = self.maxpool(x)     # 12 Maxpooling2d_3
        x = self.dropout3(x)    # 13 Dropout_3
        x = self.conv4(x)       # 14 Convolution2d_4
        x = self.elu(x)         # 15 Activation_4
        x = self.maxpool(x)     # 16 Maxpooling2d_4
        x = self.dropout4(x)    # 17 Dropout_4
        x = x.view(x.size(0), -1) # 18 Flatten_1
        x = self.dense1(x)      # 19 Dense_1
        x = self.elu(x)         # 20 Activation_5
        x = self.dropout5(x)    # 21 Dropout_5
        x = self.dense2(x)      # 22 Dense_2
        x = self.relu(x)        # 23 Activation_6
        x = self.dropout6(x)    # 24 Dropout_6
        x = self.dense(3)       # 25 Dense_3

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
