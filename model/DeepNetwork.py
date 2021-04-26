from torch import nn
import torch
import IPython.terminal.debugger as Debug


# function is used to create multiple layers perceptrons
def create_mlp(input_dim, output_dim, layer_dim, layer_num, output_act_fn, hidden_act_fn=nn.ReLU()):
    # create the input layer
    mlp = [nn.Linear(input_dim, layer_dim), hidden_act_fn]
    # create the middle layers
    for l_idx in range(layer_num - 2):
        mlp += [nn.Linear(layer_dim, layer_dim), hidden_act_fn]
    # create the output layer
    mlp += [nn.Linear(layer_dim, output_dim), output_act_fn]

    # return the mlp model
    return nn.Sequential(*mlp)


# class of deep neural network model
class DeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(DeepQNet, self).__init__()
        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, obs):
        x = self.fc_layer(obs)
        return x


class ImgEncoder(nn.Module):
    def __init__(self, obs_type='rgb'):
        super(ImgEncoder, self).__init__()

        if obs_type == 'rgb':
            self.conv_qNet = nn.Sequential(
                # 3 x 32 x 32 --> 32 x 14 x 14
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                # 32 x 14 x 14 --> 64 x 6 x 6
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # 64 x 6 x 6 --> 128 x 2 x 2
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        elif obs_type == 'depth':
            self.conv_qNet = nn.Sequential(
                # 3 x 32 x 32 --> 32 x 14 x 14
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                # 32 x 14 x 14 --> 64 x 6 x 6
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # 64 x 6 x 6 --> 128 x 2 x 2
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )

    def forward(self, x):
        # the conv2d feature size is batch_size x 512
        x = self.conv_qNet(x).view(x.shape[0], -1)

        return x


# class of the goal-conditioned network model
class GoalDeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, goal_dim, act_dim, obs_type='state'):
        # inherit from nn module
        super(GoalDeepQNet, self).__init__()

        # observation type
        self.obs_type = obs_type

        # image encoder for non-state observations
        if obs_type != 'state':
            self.encoder = ImgEncoder(obs_type)

        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, obs, goal):
        # concatenate the current observation with the goal
        if self.obs_type == 'state':
            obs_goal = torch.cat([obs, goal], dim=1)
        else:
            obs_fea = self.encoder(obs)
            goal_fea = self.encoder(goal)
            obs_goal = torch.cat([obs_fea, goal_fea], dim=1)
        # forward pass
        x = self.fc_layer(obs_goal)
        return x