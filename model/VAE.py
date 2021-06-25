"""
    Conditional VAE Model is fixed
        - Architecture:

"""
import torch
import torch.nn as nn
from model.VAELoss import VAELoss
import IPython.terminal.debugger as Debug


# Convolutional Encoder
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim, batch_size, obs_type='rgb'):
        super(CNNEncoder, self).__init__()
        self.batch_size = batch_size  # for one sample containing 4 images
        self.ori_tensor = torch.eye(4).float().to(torch.device('cuda:0'))  # encode the orientation
        # cnn layer to encode the observations
        # adding BatchNorm layer before ReLU
        in_channels = 3 if obs_type == 'rgb' else 1
        self.cnn_layer = nn.Sequential(
            # 3 x 32 x 32 --> 32 x 14 x 14
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5),
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

        # mean linear layer
        # no BatchNorm layer
        self.mu_layer = nn.Sequential(
            nn.Linear(2073, latent_dim)
        )
        # logarithm linear layer
        self.log_var_layer = nn.Sequential(
            nn.Linear(2073, latent_dim)
        )

    def forward(self, x_obs, y_map):
        # compute the visual embedding
        x_obs_feature = self.cnn_layer(x_obs)
        # flatten the features
        x_obs_flat = x_obs_feature.view(-1, 512)
        z_ori_flat = self.ori_tensor.repeat(self.batch_size, 1)
        xz_obs_ori_flat = torch.cat([x_obs_flat, z_ori_flat], dim=1).view(self.batch_size, -1)
        # combine the visual feature with orientation
        y_map_flat = y_map.view(-1, 9)
        # compute the conditional feature
        conditional_feature = torch.cat((xz_obs_ori_flat, y_map_flat), dim=1)
        # compute the mean and variance
        latent_mu = self.mu_layer(conditional_feature)
        latent_log_var = self.log_var_layer(conditional_feature)

        return latent_mu, latent_log_var, y_map_flat


# Convolutional decoder
class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, obs_type='rgb'):
        super(CNNDecoder, self).__init__()
        # dense layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 9, 1024)
        )
        # deconvolutional layer
        # adding batch norm layer except the output layer
        self.out_channels = 12 if obs_type == 'rgb' else 4
        self.de_cnn_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=self.out_channels, kernel_size=8, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # resize the input
        x = self.fc(x)
        # reshape from 2d 1024 x 1 to 3d 1024 x 1 x 1
        x = x.unsqueeze(2).unsqueeze(3)
        # deconvolution
        x = self.de_cnn_layer(x)

        return x, [x.view(-1, self.out_channels * 32 * 32),
                   torch.ones_like(x.view(-1, self.out_channels * 32 * 32))]


class CVAE(nn.Module):
    def __init__(self, latent_dim, batch_size, obs_type='rgb'):
        super(CVAE, self).__init__()

        self.batch_size = batch_size

        # create the encoder and decoder
        self.encoder = CNNEncoder(latent_dim, batch_size, obs_type)
        self.decoder = CNNDecoder(latent_dim, obs_type)

    def reparameterized(self, mu, log_var):
        # change it to be panoramic view one z for four observations
        eps = torch.randn_like(log_var)
        z = mu + torch.exp((0.5 * log_var)) * eps
        return z

    def forward(self, x_obs, y_map):
        # visual embedding
        latent_mu, latent_log_var, latent_map = self.encoder(x_obs, y_map)
        # reparameterized latent representation
        latent_z = self.reparameterized(latent_mu, latent_log_var)
        # compute the conditional latent z
        conditioned_latent_z = torch.cat((latent_z, latent_map), dim=1)
        # reconstruct the observation
        reconstructed_obs, reconstructed_obs_distribution_params = self.decoder(conditioned_latent_z)

        return reconstructed_obs, reconstructed_obs_distribution_params, [latent_mu, latent_log_var]


# # test code
# batch_size = 4
# test_obs = torch.randn(4 * batch_size, 1, 32, 32)
# test_local_map = torch.randn(batch_size, 3, 3)
#
# model = CVAE(latent_dim=32, batch_size=batch_size, obs_type='depth')
#
# reconstructed_x, distribution_x, distribution_z = model(test_obs, test_local_map)
#
# loss_func = VAELoss()
#
# loss, _, _ = loss_func(test_obs, distribution_x, distribution_z)
#
# print(loss)



