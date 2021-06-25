from torch import nn
import torch


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
        x = self.conv_qNet(x).flatten().view(x.shape[0], 128 * 2 * 2)

        return x


class PanoramaImgEncoder(nn.Module):
    def __init__(self, obs_type='panorama-rgb', mode='concat'):
        super(PanoramaImgEncoder, self).__init__()
        # observation type
        self.obs_type = obs_type
        # usage mode
        self.mode = mode
        # build the visual encoder
        if obs_type == 'panorama-rgb':
            if mode == "concat":  # we concatenate the 4 observations together
                self.concat_rgb_encoder = nn.Sequential(
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
            elif mode == "channel":  # we group the 4 observations into a group
                self.channel_rgb_encoder = nn.Sequential(
                    # 12 x 32 x 32 --> 32 x 14 x 14
                    nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5),
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
            else:
                raise Exception("Invalid encoder mode. Expect concat or channel, but get ", mode)
        elif obs_type == 'panorama-depth':  # for depth images
            if mode == "concat":
                self.concat_depth_encoder = nn.Sequential(
                    # 1 x 32 x 32 --> 32 x 14 x 14
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
            elif mode == "channel":
                self.channel_depth_encoder = nn.Sequential(
                    # 3 x 32 x 32 --> 32 x 14 x 14
                    nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5),
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
            else:
                raise Exception("Invalid encoder mode. Expect concat or channel, but get ", mode)
        else:
            raise Exception("Invalid encoder observation. Expect panorama-rgb or panorama-depth, but get ", obs_type)

    def forward(self, x):
        if self.obs_type == 'panorama-rgb':
            if self.mode == "concat":  # we concatenate the 4 observations together
                x = self.concat_rgb_encoder(x).reshape(-1, 128 * 2 * 2 * 4)
            elif self.mode == "channel":  # we group the 4 observations into a group
                x = x.reshape(-1, 12, 32, 32)  # merge 4 observations into one with shape (12 x W x H)
                x = self.channel_rgb_encoder(x).reshape(-1, 128 * 2 * 2)
            else:
                raise Exception("Invalid encoder mode. Expect concat or channel, but get ", self.mode)
        elif self.obs_type == 'panorama-depth':  # for depth images
            if self.mode == "concat":
                x = self.concat_depth_encoder(x).reshape(-1, 128 * 2 * 2 * 4)
            elif self.mode == "channel":
                x = x.reshape(-1, 4, 32, 32)
                x = self.channel_depth_encoder(x).reshape(-1, 128 * 2 * 2)
            else:
                raise Exception("Invalid encoder mode. Expect concat or channel, but get ", self.mode)
        else:
            raise Exception("Invalid encoder observation. Expect panorama-rgb or panorama-depth, but get ",
                            self.obs_type)

        return x


# class of the goal-conditioned network model
class GoalDeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_type='state', panorama_obs_mode='concat', act_dim=4):
        # inherit from nn module
        super(GoalDeepQNet, self).__init__()

        # observation type
        self.obs_type = obs_type

        # panorama observation mode
        self.panorama_obs_mode = panorama_obs_mode

        # dimension of the action
        self.act_dim = act_dim

        # image encoder for non-state observations
        if obs_type == 'rgb' or obs_type == 'depth':
            self.encoder = ImgEncoder(obs_type)
            # for debug
            obs_dim = 512
            goal_dim = 512
        # image encoder for panorama observations
        elif obs_type == "panorama-rgb" or obs_type == "panorama-depth":
            self.encoder = PanoramaImgEncoder(obs_type, panorama_obs_mode)
            # for debug
            if panorama_obs_mode == "concat":
                obs_dim = 2048
                goal_dim = 2048
            else:
                obs_dim = 512
                goal_dim = 512
        else:
            # for debug case
            obs_dim = 4  # [x, y, sin(theta), cos(theta)]
            goal_dim = 4  # [x, y, sin(theta), cos(theta)]

        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.act_dim),
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


# create controller
my_controller = GoalDeepQNet(obs_type='state', panorama_obs_mode='concat', act_dim=3)
my_controller.load_state_dict(torch.load('./test_model_binary.pt', map_location='cpu'))

# create test state and goal
action_names = ['turn_left', 'turn_right', 'forward']
test_state = torch.tensor([0.5, 1.0, 0, 1.0]).view(1, 4)
test_goal = torch.tensor([3.5, 1.0, 0, 1.0]).view(1, 4)

# prediction action
action = my_controller(test_state, test_goal).max(dim=1)[1].item()

print(action_names[action])

# # save the results again
# torch.save(my_controller.state_dict(), './test_model_binary.pt', _use_new_zipfile_serialization=False)
