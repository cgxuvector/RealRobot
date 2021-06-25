from model import VAELoss, VAE
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import tqdm
import json
import numpy as np
import random
import matplotlib.pyplot as plt

import IPython.terminal.debugger as Debug


# Todo: vae does not make sense.


def load_data(m_size, idx_list):
    data = []
    for idx in idx_list:
        with open(f"./data/vae/vae_data_maze_{m_size}_{idx}.json", 'r') as f_in:
            files = json.load(f_in)
        f_in.close()

        for room in files:
            m_data = [[loc['local map'], loc['depth observation']] for loc in room]
            data += m_data

        print(f"maze id {idx}: room number {len(files)}, total locations {len(files) * 20}")

    return data


def sample_batch_data(dataset, batch_size, device):
    assert batch_size < len(dataset), "Batch size is too big."

    sampled_data = random.sample(dataset, batch_size)

    local_map = []
    depth_obs = []
    tmp = [[local_map.append(d[0]), depth_obs.append(d[1])] for d in sampled_data]

    return [torch.from_numpy(np.array(depth_obs)).permute(0, 1, 4, 2, 3).view(-1, 1, 32, 32).float().to(device),
            torch.from_numpy(np.array(local_map)).float().to(device)]


def cvae_eval():
    device = torch.device('cuda:0')
    eval_model = VAE.CVAE(latent_dim=64,
                          batch_size=8,
                          obs_type='depth').to(device)
    eval_model.load_state_dict(torch.load('./test_cvae_model.pt'))
    eval_model.eval()
    decoder = eval_model.decoder

    local_map = torch.tensor(np.array([[1, 1, 1],
                                       [1, 0, 0],
                                       [1, 0, 1]])).view(-1,9).float().to(device)

    z = torch.randn(1, 64).to(device)
    z_map_cat = torch.cat([z, local_map], dim=1)

    reconstruct_x, _ = decoder(z_map_cat)


def train_vae(configs):
    # start tensorboard
    current_time = datetime.today()
    log_dir_path = os.path.join(configs['log_dir'],
                                configs['run_model'],
                                configs['run_label'],
                                current_time.strftime("%m-%d"),
                                current_time.strftime("%H-%M-%S") +
                                f"_{configs['run_label']}_batch_{configs['batch_size']}")

    tb = SummaryWriter(log_dir=log_dir_path)

    # create device
    device = torch.device(configs['device'])

    # create models
    model = VAE.CVAE(latent_dim=configs['latent_dim'],
                     batch_size=configs['batch_size'],
                     obs_type='depth').to(device)
    loss_func = VAELoss.VAELoss()

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=configs['lr'])

    # start training
    ebar = tqdm.trange(configs['epoch'], desc='Epoch bar')
    # epoch iteration
    for e in ebar:
        # shuffle the buffer
        random.shuffle(trn_data)
        ibar = tqdm.trange(configs['iteration'], desc='Iteration bar')
        # batch iteration
        for it in ibar:
            # sample a mini-batch
            batch_data = sample_batch_data(configs['dataset'], configs['batch_size'], device)

            # forward pass
            _, distribution_x, distribution_z = model(batch_data[0], batch_data[1])

            # compute the loss
            loss, log_x_loss, kl_loss = loss_func(batch_data[0], distribution_x, distribution_z)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to tensorboard
            tb.add_scalar("Total loss", loss.item(), e * configs['iteration'] + it)
            tb.add_scalar("Recon loss", log_x_loss.item(), e * configs['iteration'] + it)
            tb.add_scalar("KL loss", kl_loss.item(), e * configs['iteration'] + it)

        torch.save(model.state_dict(), f"./test_cvae_model.pt")


if __name__ == "__main__":
    # load training data
    maze_size = 7
    maze_idx_list = [0]
    trn_data = load_data(maze_size, maze_idx_list)

    # training configurations
    trn_configs = {
        'dataset': trn_data,

        'batch_size': 8,
        'lr': 1e-4,

        'epoch': 100,
        'iteration': 1000,

        'latent_dim': 64,

        'device': 'cuda:0',

        'run_model': 'cvae',
        'run_label': 'cvae_test_run',
        'log_dir': './runs'
    }

    # run training
    # train_vae(trn_configs)
    cvae_eval()

