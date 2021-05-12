import torch
from torch import nn
import torch.nn.functional as F


class NonLocal2DBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocal2DBlock, self).__init__()

        # normalize C
        self.C = 8

        # define the f(x) embedding
        self.f_func = nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels//8 if in_channels > self.C else in_channels,
                                kernel_size=1)
        # define the g(x) embedding
        self.g_func = nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels//8 if in_channels > self.C else in_channels,
                                kernel_size=1)
        # define the h(x) embedding
        self.h_func = nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=1)

        # define the SoftMat layer
        self.soft_max_func = nn.Softmax(dim=-1)

        # define a learnable parameter gamma to regularize the combination
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # get the size of input tensor batch
        b_size, c_dim, w, h = x.size()

        # compute the f embedding feature and transpose it for computing attention map
        # Explanation:
        #   Suppose x has the shape (1, 16, W, H), in other words, there is only one sample in the batch.
        #   For a pixel at position (i, j), we denote it as p_ij. After the performing the embedding f(x),
        #   suppose we keep the channels to be the same (i.e., 16), then we have 16 feature maps with shape W x N.
        #   We can think about a feature representation for one pixel in the image, which is using the values in the
        #   16 feature maps to form a vector. To do this, we can flatten all the 16 feature maps and form a 2-D tensor
        #   with shape 16 x N where N = W x H. Therefore, each column is a feature of one pixel in the image.
        f_x = self.f_func(x).view(b_size, -1, w*h).permute(0, 2, 1)  # B x N x C_reduced  (where N = W * H)
        # compute the g embedding feature
        g_x = self.g_func(x).view(b_size, -1, w*h)  # B x C_reduced x N
        # compute the h embedding feature
        h_x = self.h_func(x).view(b_size, -1, w*h)  # B x C x N
        # compute the attention map
        # Explanation:
        #    If we consider dot-similarity, then we have to compute the similarity between i and all other possible j.
        #    With the feature representation above, we can compute a similarity matrix with shape N x N.
        attention_map = self.soft_max_func(torch.bmm(f_x, g_x))  # B x N x N
        # compute the attended feature map
        # Explanation:
        #     Here is how the non-local equation works.
        attended_x = torch.bmm(h_x, attention_map)  # B x C x N

        # reshape the attended feature map
        attended_x = attended_x.view(b_size, c_dim, w, h)

        # combine the attended feature map and the original feature map
        combined_attended_x = x + self.gamma * attended_x

        return combined_attended_x


class SelfAttendedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super(SelfAttendedConv2d, self).__init__()

        # define the attention layer
        self.self_attention_layer = NonLocal2DBlock(in_channels=in_channels)

        # define the conv2d layer
        self.conv2d_layer = nn.Sequential(
            # 1x32x32 -> 3x32x32
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, stride=stride),
            # 3x32x32 -> 3x16x16
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        # attending the input
        attended_x = self.self_attention_layer(x)

        # feeding into convolutional layers
        x = self.conv2d_layer(attended_x)

        return x, attended_x


# UNet based hard attention model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # define down-sampling block
        # 1 x 32 x 32 -> 32 x 16 x 16
        self.down_sampling_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 32 x 16 x 16 -> 64 x 8 x 8
        self.down_sampling_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 64 x 8 x 8 -> 128 x 4 x 4
        self.down_sampling_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=128, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 128 x 4 x 4 -> 256 x 2 x 2
        self.down_sampling_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=256, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # define bottle-neck layer
        self.bottle_neck_fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 2 * 2),
            nn.ReLU()
        )

        # define up-sampling block
        # 256 x 2 x 2 -> 128 x 4 x 4
        self.up_sampling_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=2, stride=2),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU()
        )
        # 128 x 4 x 4 -> 64 x 8 x 8
        self.up_sampling_block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU()
        )
        # 64 x 8 x 8 -> 32 x 16 x 16
        self.up_sampling_block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2),
            nn.InstanceNorm2d(num_features=32),
            nn.ReLU()
        )
        # 32 x 32 x 32 -> 1 x 32 x 32
        self.up_sampling_block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=2, stride=2),
            nn.InstanceNorm2d(num_features=1),
            nn.ReLU()
        )

        # output
        # self.output = nn.Softmax(dim=1)

    def forward(self, s, m):
        # batch size
        b_size = s.shape[0]
        # down-sampling process
        x = self.down_sampling_block1(m)
        skip1 = x
        x = self.down_sampling_block2(x)
        skip2 = x
        x = self.down_sampling_block3(x)
        skip3 = x
        x = self.down_sampling_block4(x)
        skip4 = x

        # bottle next
        x = torch.cat([s, x.reshape(b_size, -1)], dim=1)
        x = self.bottle_neck_fc(x).reshape(b_size, 256, 2, 2)

        # up-sampling process
        x = torch.cat([x, skip4], dim=1)
        x = self.up_sampling_block1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_sampling_block2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_sampling_block3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_sampling_block4(x)

        return x
        #
        # x = x.reshape(b_size, -1)
        # x_prob = self.output(x).reshape(b_size, 1, 32, 32)

        # return x_prob

#
# import numpy as np
# test_state = torch.randn(1, 2)
# test_map = torch.randn(1, 1, 32, 32)
#
# model = UNet()
#
# output = model(test_state, test_map).squeeze(dim=0).squeeze(dim=0)
#
# output_arr = output.detach().numpy()










