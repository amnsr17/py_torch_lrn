import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # input, convolutional features required, window/kernel size
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # To generate the distribution of predictions we need to
        # go from Conv2d to Linear layers

        # -- For input size of the Linear layer passing random data through Conv layers
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2)

    def convs(self, x):
        """ A forward method for passing data through convolutional layers.
            Also serves to check the shape of the tensor at the output of the
            convolutional layers by passing random data
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        print(x[0].shape)
        if self._to_linear is None:
            # getting the shape of the data after passing through the convolutional layers
            # x is coming as batch of data so x[0] is the first element in the batch
            # shape returns a tuple of 3 elements. First element of shape tuple * second element of shape tuple * third element of shape tuple
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        # pass data through convolutional layers
        x = self.convs(x)
        # reshape it for Linear layers
        x = x.view(-1, self._to_linear)
        # Now passing thrugh the linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)




# Loading the labeled data
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

net = Network()
# net.forward(training_data)



