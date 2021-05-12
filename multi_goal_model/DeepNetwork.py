from torch import nn
import torch
import torch.nn.functional as F
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


""" Define models for RL agent
"""


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


# class of the goal-conditioned network model
class GoalDeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, goal_dim, act_dim):
        # inherit from nn module
        super(GoalDeepQNet, self).__init__()
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
        obs_goal = torch.cat([obs, goal], dim=1)
        # forward pass
        x = self.fc_layer(obs_goal)
        return x


""" Define the two networks for the two models method
"""


class InverseDynModel(nn.Module):
    def __init__(self, state_dim, output_dim, hidden_dim=256, hidden_layers=3):
        super(InverseDynModel, self).__init__()

        # state encoder
        self.state_encoder = nn.Sequential(nn.Identity())

        # fully connected layers
        self.fc = create_mlp(state_dim * 2, output_dim, hidden_dim, 4, nn.Identity(), nn.ReLU())

    def forward(self, state, next_state):
        # compute the representation
        state_repr = self.state_encoder(state).float()
        next_state_repr = self.state_encoder(next_state).float()
        # concatenate together
        repr_concat = torch.cat([state_repr, next_state_repr], dim=1)
        # forward
        x = self.fc(repr_concat)
        return x


# current for state (i.e., [x, y])
class ForwardDynModel(nn.Module):
    def __init__(self, state_dim, goal_dim, map_dim, output_dim, hidden_dim=256, hidden_layers=3):
        super(ForwardDynModel, self).__init__()

        # state encoder
        self.state_encoder = nn.Sequential(nn.Identity())

        # goal encoder
        self.goal_encoder = nn.Sequential(nn.Identity())

        # map encoder
        self.map_encoder = nn.Sequential(nn.Identity())

        # fully connected layers
        self.fc = create_mlp(state_dim + goal_dim + map_dim, output_dim, hidden_dim, hidden_layers + 1, nn.ReLU(),
                             nn.ReLU())

    def forward(self, state, g, m):
        # compute the representation
        state_repr = self.state_encoder(state).float()  # 1x2
        goal_repr = self.goal_encoder(g).float()  # 1x2
        map_repr = self.map_encoder(m).view(g.shape[0], -1)  # 1x64
        # concatenate the representation
        repr_concat = torch.cat([state_repr, goal_repr, map_repr], dim=1)  # 2 + 64 * 2
        # forward
        x = self.fc(repr_concat)
        return x


""" Define the predictor model directly
"""


class SLActionModel(nn.Module):
    """
        A simple supervised baseline that predicts the action from state, goal, and map
    """

    def __init__(self, state_dim, act_dim, goal_dim, m_dim):
        super(SLActionModel, self).__init__()

        self.state_encoder = nn.Sequential(nn.Identity())
        self.goal_encoder = nn.Sequential(nn.Identity())
        self.map_encoder = nn.Sequential(nn.Identity())
        self.fc = create_mlp(state_dim + goal_dim + m_dim, act_dim, 256, 3, nn.Softmax(dim=1), nn.ReLU())

    def forward(self, state, goal, m):
        s_repr = self.state_encoder(state).float()
        g_repr = self.goal_encoder(goal).float()
        m_repr = self.map_encoder(m).view(state.shape[0], -1).float()
        repr_cancat = torch.cat([s_repr, g_repr, m_repr], dim=1)
        x = self.fc(repr_cancat)
        return x


""" Define the deterministic dynamic models.
    It learns two functions:
        - Transition function: P_theta(s_t+1 | s_t, a_t, map)
        - Reward function: R_rho(r_t | s_t, a_t, g)

    However, the limitation of this algorithm is
        - one-step transition function
        - accumulative errors
        - deterministic
"""


class MapEncoder(nn.Module):
    def __init__(self, map_feature='flatten'):
        """ The map encoder will indicate the map in two fashions:
                - A flatten representation
                - A conv2d representation: the input has to be 1x32x32
        """
        super(MapEncoder, self).__init__()

        # encoder type
        self.mode = map_feature

        # convolutional map encoder
        self.conv_encoder = nn.Sequential(
            # 1x32x32 -> 3x32x32
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, dilation=1, stride=1),
            # 3x32x32 -> 3x16x16
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # 3x16x16 -> 32x16x16
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, dilation=1, stride=1),
            # 32x16x16 -> 32x8x8
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # 32x8x8 -> 64x8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=1, stride=1),
            # 64x8x8 -> 64x4x4
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # 64x4x4 -> 128x4x4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, dilation=1, stride=1),
            # 128x4x4 -> 128x2x2
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, m):
        if self.mode == "conv2d":
            x = self.conv_encoder(m).view(m.shape[0], -1)
        else:
            x = m.view(m.shape[0], -1)
        return x


class SelfAttendedMapEncoder(nn.Module):
    def __init__(self, map_feature='flatten'):
        super(SelfAttendedMapEncoder, self).__init__()

        # encoder type
        self.mode = map_feature

        # create the attended convolutional layers
        self._layer_1 = SelfAttendedConv2d(in_channels=1, out_channels=3, kernel_size=3)
        self._layer_2 = SelfAttendedConv2d(in_channels=3, out_channels=32, kernel_size=3)
        self._layer_3 = SelfAttendedConv2d(in_channels=32, out_channels=64, kernel_size=3)
        self._layer_4 = SelfAttendedConv2d(in_channels=64, out_channels=128, kernel_size=3)

    def forward(self, m):
        if self.mode == "conv2d":
            x, _ = self._layer_1(m)
            x, _ = self._layer_2(x)
            x, _ = self._layer_3(x)
            x, _ = self._layer_4(x)
            x = x.view(m.shape[0], -1)
        else:
            x = m.view(m.shape[0], -1)
        return x


class DeterministicDynaModel(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, map_fea_dim, map_feature="flatten", self_attention=False):
        super(DeterministicDynaModel, self).__init__()
        # type of the map feature used in the dynamic model
        self.map_feature = map_feature

        # map encoder
        if map_feature == "conv2d":
            if self_attention:
                self.map_encoder = SelfAttendedMapEncoder(map_feature)
            else:
                self.map_encoder = MapEncoder(map_feature)

        # transition function
        self.trans_func = nn.Sequential(
            nn.Linear(state_dim + action_dim + map_fea_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

        # reward function
        self.reward_func = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # we use sparse rewards, we use classification for better performance
        )

    def forward(self, state, action, goal, m):
        # reshape the input
        state_tensor = state.view(state.shape[0], -1)
        action_tensor = action.view(state.shape[0], -1)
        goal_tensor = goal.view(state.shape[0], -1)
        # output the map feature based on type
        if self.map_feature == "conv2d":
            map_tensor = self.map_encoder(m)
        else:
            map_tensor = m.view(state.shape[0], -1)
        # concatenate
        state_action_map = torch.cat([state_tensor, action_tensor, map_tensor], dim=1)
        state_action_goal = torch.cat([state_tensor, action_tensor, goal_tensor], dim=1)
        # predict the next state
        next_state_x = self.trans_func(state_action_map)
        # predict the reward
        reward_x = self.reward_func(state_action_goal)

        return next_state_x, reward_x


""" Define the deterministic dynamics models using hyper-network
    We will still learn P_theta(s_t+1 | s_t, a_t)
    we also learn theta = f(m)
"""


class HyperNetwork(nn.Module):
    def __init__(self):
        super(HyperNetwork, self).__init__()
        # convolutional encoder
        self.conv_layer = MapEncoder('conv2d')

        # generate the weight for the first layer
        # add non-linearity
        # self.w1_fc = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 3 * 256)
        # )
        # for ego centric action
        self.w1_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * 256)
        )

        # generate the bias for the first layer
        self.b1_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # generate the weight for the second layer
        self.w2_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 256)
        )

        # generate the bias for the second layer
        self.b2_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # generate the weight for the second layer
        # self.w3_fc = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256 * 2)
        # )
        self.w3_fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 3)
        )

        # generate the bias for the second layer
        # self.b3_fc = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2)
        # )
        # for egocentric data
        self.b3_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, m):
        # generate the map embedding
        x = self.conv_layer(m).view(m.shape[0], -1)
        # generate the weights

        # w1 = self.w1_fc(x).view(256, 3)
        # b1 = self.b1_fc(x).view(256,)
        # w2 = self.w2_fc(x).view(256, 256)
        # b2 = self.b2_fc(x).view(256,)
        # w3 = self.w3_fc(x).view(2, 256)
        # b3 = self.b3_fc(x).view(2,)

        b_size = m.shape[0]
        # change the dimension for egocentric data
        # w1 = self.w1_fc(x).view(b_size, 3, 256)
        w1 = self.w1_fc(x).view(b_size, 4, 256)
        b1 = self.b1_fc(x).view(b_size, 1, 256)
        w2 = self.w2_fc(x).view(b_size, 256, 256)
        b2 = self.b2_fc(x).view(b_size, 1, 256)
        # w3 = self.w3_fc(x).view(b_size, 256, 2)
        # b3 = self.b3_fc(x).view(b_size, 1, 2)
        w3 = self.w3_fc(x).view(b_size, 256, 3)
        b3 = self.b3_fc(x).view(b_size, 1, 3)

        # return the weights
        return [w1, b1, w2, b2, w3, b3]


class DynamicModel(nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.dyna_model = None

    # def forward(self, state, action, weights):
    #     self.dyna_model = "hyper-dynamic-model"
    #     state = state.view(state.shape[0], -1)
    #     action = action.view(action.shape[0], -1)
    #     # state goal representation
    #     state_action = torch.cat([state, action], dim=1)
    #     # get all the weights
    #     w1, b1, w2, b2, w3, b3 = weights
    #     # compute the output
    #     # the current dynamic model is linear
    #     x = F.linear(state_action, w1, b1)  # layer one
    #     x = F.relu(x)
    #     x = F.linear(x, w2, b2)  # layer two
    #     x = F.relu(x)
    #     x = F.linear(x, w3, b3)  # layer there
    #
    #     return x

    def forward(self, state, action, weights):
        self.dyna_model = "hyper-dynamic-model"
        state = state.view(state.shape[0], 1, -1)
        action = action.view(action.shape[0], 1, -1)
        # state goal representation
        state_action = torch.cat([state, action], dim=2)
        # get all the weights
        w1, b1, w2, b2, w3, b3 = weights
        # compute the output
        # the current dynamic model is linear
        x = torch.matmul(state_action, w1) + b1  # layer one
        x = F.relu(x)
        x = torch.matmul(x, w2) + b2  # layer two
        x = F.relu(x)
        x = torch.matmul(x, w3) + b3  # layer there

        # squeeze the x
        x = x.squeeze(dim=1)

        return x


""" Dynamic filter network implementation
        - Dynamic convolution
"""


class DynaFilterNet(nn.Module):
    def __init__(self, m_size):
        super(DynaFilterNet, self).__init__()

        # map size
        self.m_size = m_size

        # define the dynamic convolution
        self.fc_layer = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, m_size ** 2)
        )

    def forward(self, s):
        # get the batch size
        b_size = s.shape[0]
        # generate the mask
        mask = self.fc_layer(s).view(b_size, self.m_size, self.m_size)

        return mask
