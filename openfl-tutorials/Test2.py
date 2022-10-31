import openfl.native as fx
from openfl.federated import FederatedModel, FederatedDataSet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import openfl.interface.aggregation_functions as agg

import openfl.interface.attackers as att

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(0)
np.random.seed(0)

fx.init()

NUM_COLLABORATORS = 10

collaborator_list = [str(i) for i in range(NUM_COLLABORATORS)]
fx.init('torch_cnn_mnist', col_names=collaborator_list)

fx.update_plan({"aggregator.settings.rounds_to_train": 5})

def one_hot(labels, classes):
    return np.eye(classes)[labels]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

train_images, train_labels = trainset.train_data, np.array(trainset.train_labels)
train_images = torch.from_numpy(np.expand_dims(train_images, axis=1)).float()
train_labels = one_hot(train_labels, 10)

validset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

valid_images, valid_labels = validset.test_data, np.array(validset.test_labels)
valid_images = torch.from_numpy(np.expand_dims(valid_images, axis=1)).float()
valid_labels = one_hot(valid_labels, 10)
#%%
feature_shape = train_images.shape[1]
classes = 10

fl_data = FederatedDataSet(train_images, train_labels, valid_images, valid_labels, batch_size=32, num_classes=classes)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

#optimizer = lambda x: optim.Adam(x, lr=1e-4)
optimizer = lambda x: optim.SGD(x, lr=1e-4)

def cross_entropy(output, target):
    """Binary cross-entropy metric
    """
    return F.binary_cross_entropy_with_logits(input=output, target=target)

fl_model = FederatedModel(build_model=Net, optimizer=optimizer, loss_fn=cross_entropy, data_loader=fl_data)

attacker = att.MinMaxAttacker(f=2)

fl_data = FederatedDataSet(train_images, train_labels, valid_images, valid_labels, batch_size=32, num_classes=classes)

experiment_collaborators = {col_name:col_model for col_name, col_model in zip(collaborator_list, fl_model.setup(len(collaborator_list)))}


final_fl_model = fx.run_experiment(experiment_collaborators, {
    'tasks.train.aggregation_type': agg.Flame(delta=0.001, epsilon=1000, attacker=attacker),
})

#final_fl_model = fx.run_experiment(experiment_collaborators)