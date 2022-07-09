import torch.nn as nn
"""
I HAVE NO IDEA WHETHER IT WORKS OR NOT WHEN I AM UNSING RNN WITH A TANH ON ADVANTAGE AND VALUE FUNCATION.
CREATED BY SIYUEXI
2022.07.02
"""
class DQN(nn.Module):
    # image size is fixed at 84*84
    def __init__(self,n_actions, n_states) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(n_states, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        y = self.fc2(x)
        return y


class DuelingDQN(nn.Module):
    def __init__(self, n_actions, n_states):
        super().__init__()
        self.n_actions = n_actions
        
        self.conv1 = nn.Conv2d(n_states, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc1_adv = nn.Linear(7*7*64, 512)
        self.fc1_val = nn.Linear(7*7*64, 512)

        self.fc2_adv = nn.Linear(512, n_actions)
        self.fc2_val = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        a = self.relu(self.fc1_adv(x))
        v = self.relu(self.fc1_val(x))

        a = self.fc2_adv(a)
        v = self.fc2_val(v)
        
        y = a + v.expand(x.size(0), self.n_actions) - a.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)

        return y


class CRD2QN(nn.Module):
    # image size is fixed at 84*84
    def __init__(self,n_actions, n_states) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        # cnn for single feature extraction
        self.conv1 = nn.Conv2d(n_actions, 32, 8, 4, groups=n_actions)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, groups=n_actions)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, groups=n_actions)
        # rnn for multiple feature concatenation
        self.recu1 = nn.RNN(7*7*16, n_states, 1, batch_first=True)
        self.recu2 = nn.RNN(7*7*16, n_states, 1, batch_first=True)
        # fc for prediction
        self.fc1 = nn.Linear(n_states, n_states)
        self.fc2 = nn.Linear(n_states, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), self.n_states,-1)

        _,a = self.recu1(x)
        _,v = self.recu2(x)

        a = a[0,:,:].view(x.size(0),-1)
        v = v[0,:,:].view(x.size(0),-1)

        a = self.fc1(a)
        v = self.fc2(v)

        y = a + v.expand(x.size(0), self.n_actions) - a.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)

        return y


def get_net(type, n_actions, n_states, ddqn_model):
    
    if type == "CRD2QN":
        net = CRD2QN(n_actions, n_states)
    elif type == "DuelingDQN":
        net = DuelingDQN(n_actions, n_states)
    else:
        net = DQN(n_actions, n_states)
    
    if ddqn_model:
        sub_net = net

        return net, sub_net

    return net
    

"""UNIT TESTING"""
# import torch
# x = torch.randn([2,4,84,84])
# net = get_net()
# y = net(x)
# print(y.shape)

# net_1, net_2 = get_net(ddqn=True)
# y1 = net_1(x)
# y2 = net_2(x)
# print(y1.shape)
# print(y2.shape)
