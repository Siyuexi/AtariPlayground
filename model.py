import torch.nn as nn
"""
CRD2QN NEED ALL STEP'S VALUE FOR TRAINING. WHEN USING IT, THE MEANING OF N_STATES IS DIFFERENT.
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


class D2QN(nn.Module):
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
        self.conv1 = nn.Conv2d(n_states, 32, 8, 4, groups=n_states)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, groups=n_states)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, groups=n_states)
        # rnn for multiple feature concatenation
        self.recu1 = nn.GRU(int(7*7*64 / n_actions), 128, 1, batch_first=True)
        self.recu2 = nn.GRU(int(7*7*64 / n_actions), 128, 1, batch_first=True)
        # mlp for dimension reduction and activation
        self.fc1 = nn.Linear(128, n_actions)
        self.fc2 = nn.Linear(128, 1)
        # relu
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        a = x.view(x.size(0), self.n_states, -1)
        v = x.view(x.size(0), self.n_states, -1)

        _,a = self.recu1(a)
        _,v = self.recu2(v)

        a = a[0,:,:].view(x.size(0),-1)
        v = v[0,:,:].view(x.size(0),-1)
        
        a = self.fc1(a)
        v = self.fc2(v)

        y = a + v.expand(x.size(0), self.n_actions) - a.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)

        return y


def get_net(type, n_actions, n_states):
    
    if type == "CRD2QN":
        net = CRD2QN(n_actions, n_states)
        sub_net = CRD2QN(n_actions, n_states)
        print("CRD2QN")
    elif type == "D2QN":
        net = D2QN(n_actions, n_states)
        sub_net = D2QN(n_actions, n_states)
        print("D2QN")
    else:
        net = DQN(n_actions, n_states)
        sub_net = DQN(n_actions, n_states)
        print("DQN")

    return net, sub_net

    

"""UNIT TESTING"""
# import torch
# x = torch.randn([2,4,84,84])
# net_1, net_2 = get_net("CRD2QN", 4, 4)
# y1 = net_1(x)
# y2 = net_2(x)
# print(y1.shape)
# print(y2.shape)
