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


""" Define the two networks separately
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


class ForwardDynModel(nn.Module):
    def __init__(self, state_dim, goal_dim, map_dim, output_dim, hidden_dim=256, hidden_layers=3):
        super(ForwardDynModel, self).__init__()

        # state encoder
        self.state_encoder = nn.Sequential(nn.Identity())

        # goal encoder
        self.goal_encoder = nn.Sequential(
            # (1x32x32 -> 2x16x16)
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, padding=0, stride=2, dilation=1),
            nn.ReLU(),
            # (2x16x16 -> 4x8x8)
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, padding=0, stride=2, dilation=1),
            nn.ReLU(),
            # (4x8x8 -> 4x4x4)
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, padding=0, stride=2, dilation=1),
            nn.ReLU()
        )

        # fully connected layers
        self.fc = create_mlp(state_dim + goal_dim + map_dim, output_dim, hidden_dim, 4, nn.ReLU(), nn.ReLU())

    def forward(self, state, g, m):
        # compute the representation
        state_repr = self.state_encoder(state).float()  # 1x2
        goal_repr = self.goal_encoder(g).view(g.shape[0], -1)  # 1x64
        map_repr = self.goal_encoder(m).view(g.shape[0], -1)  # 1x64
        # concatenate the representation
        repr_concat = torch.cat([state_repr, goal_repr, map_repr], dim=1)  # 2 + 64 * 2
        # forward
        x = self.fc(repr_concat)
        return x


class ForwardDynModelRaw(nn.Module):
    def __init__(self, state_dim, goal_dim, map_dim, output_dim, hidden_dim=256, hidden_layers=3):
        super(ForwardDynModelRaw, self).__init__()

        # state encoder
        self.state_encoder = nn.Sequential(nn.Identity())

        # goal encoder
        self.goal_encoder = nn.Sequential(nn.Identity())

        # map encoder
        self.map_encoder = nn.Sequential(nn.Identity())

        # fully connected layers
        self.fc = create_mlp(state_dim + goal_dim + map_dim, output_dim, hidden_dim, 4, nn.ReLU(), nn.ReLU())

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
        A simple supervised baseline
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
    def __init__(self, feature_mode='flatten'):
        super(MapEncoder, self).__init__()

        # encoder type
        self.mode = feature_mode

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


class DeterministicDynaModel(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, map_dim=512, map_feature="flatten"):
        super(DeterministicDynaModel, self).__init__()

        # map encoder
        self.map_encoder = MapEncoder(map_feature)

        # transition function
        self.trans_func = nn.Sequential(
            nn.Linear(state_dim + action_dim + map_dim, 256),
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
        map_tensor = self.map_encoder(m)
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

# how to implement the Hyper-network?


"""Training the prediction model directly on state and goal is not efficient to predict the correct next state
    Here I want to leverage the power of VAE model:
    P(s_{t+1}, z| s_t, a_t) = P(s_{t+1} | z)P(z | s_t, a_t). I will use P(z | s_t, g) to estimate P(z | s_{t}, a_{t})
"""


class SAEncoder(nn.Module):
    def __init__(self):
        super(SAEncoder, self).__init__()
        # parameters
        self.state_dim = 2
        self.act_dim = 1
        self.map_dim = 225
        self.latent_dim = 8

        # feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.act_dim + self.map_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # mean output layer
        self.mu_layer = nn.Sequential(
            nn.Linear(256, self.latent_dim)
        )

        # log sigma output layer
        self.log_sigma_layer = nn.Sequential(
            nn.Linear(256, self.latent_dim),
            nn.Softplus()
        )

    def forward(self, s, a, m):
        # concatenate the state, action and map
        s_repr = s.view(-1, 2)
        a_repr = a.view(-1, 1)
        m_repr = m.view(s.shape[0], -1)
        s_a_m_repr = torch.cat([s_repr, a_repr, m_repr], dim=1)
        # compute the feature
        feature_repr = self.encoder(s_a_m_repr)
        # compute the mean
        latent_mu = self.mu_layer(feature_repr)
        # compute the log variance
        latent_log_sigma = self.log_sigma_layer(feature_repr)

        return [latent_mu, latent_log_sigma]


class SGEncoder(nn.Module):
    def __init__(self):
        super(SGEncoder, self).__init__()
        # parameters
        self.state_dim = 2
        self.goal_dim = 2
        self.map_dim = 225
        self.latent_dim = 8

        # feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim + self.map_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # mean output layer
        self.mu_layer = nn.Sequential(
            nn.Linear(256, self.latent_dim)
        )

        # log sigma output layer
        self.log_sigma_layer = nn.Sequential(
            nn.Linear(256, self.latent_dim),
            nn.Softplus()
        )

    def forward(self, s, g, m):
        # concatenate the state, action and map
        s_repr = s.view(-1, 2)
        g_repr = g.view(-1, 2)
        m_repr = m.view(s.shape[0], -1)
        s_g_m_repr = torch.cat([s_repr, g_repr, m_repr], dim=1)
        # compute the feature
        feature_repr = self.encoder(s_g_m_repr)
        # compute the mean
        latent_mu = self.mu_layer(feature_repr)
        # compute the variance
        latent_log_sigma = self.log_sigma_layer(feature_repr)

        return [latent_mu, latent_log_sigma]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # parameters
        self.state_dim = 2
        self.latent_dim = 8

        # define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, z):
        # decode the latent representation
        x = self.decoder(z)
        return x


class VAEForwardModel(nn.Module):
    def __init__(self):
        super(VAEForwardModel, self).__init__()
        # create the state-action encoder
        self.sa_encoder = SAEncoder()
        # create the state-goal encoder
        self.sg_encoder = SGEncoder()
        # create the decoder
        self.predictor = Decoder()

    def reparameterize_func(self, mu, log_var):
        # compute randomness
        eps = torch.rand_like(log_var)  # sample a random eps ~ N(0, 1) with the same shape
        # compute std
        std = torch.exp(0.5 * log_var)
        # sample the z
        z = mu + std * eps
        return z

    def forward(self, s, a, g, m):
        # pass the state-action encoder
        sa_latent_mu, sa_latent_log_var = self.sa_encoder(s, a, m)
        # pass the state-goal encoder
        sg_latent_mu, sg_latent_log_var = self.sg_encoder(s, g, m)

        # compute the latent z from sa_mu and sa_var
        latent_z = self.reparameterize_func(sa_latent_mu, sa_latent_log_var)

        # predict the next state
        pred_next_s = self.predictor(latent_z)

        return pred_next_s, [sa_latent_mu, sa_latent_log_var], [sg_latent_mu, sg_latent_log_var]



