from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import os
import struct
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from skimage.transform import resize

use_gpu = torch.cuda.is_available()

# training function used in pretraining
def run_epoch(train, model, exp_data, clip, optimizer=None, batch_size=64, num_missing=None, teacher_forcing=True):
    losses = []
    inds = np.random.permutation(exp_data.shape[0])
    
    i = 0
    while i + batch_size <= exp_data.shape[0]:
        ind = torch.from_numpy(inds[i:i+batch_size]).long()
        i += batch_size
        data = exp_data[ind]
    
        if use_gpu:
            data = data.cuda()

        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.squeeze().transpose(0, 1))
        ground_truth = data.clone()
        if num_missing is None:
            #num_missing = np.random.randint(data.shape[0] * 18 // 20, data.shape[0])
            num_missing = np.random.randint(data.shape[0] * 4 // 5, data.shape[0])
            #num_missing = 40
        missing_list = torch.from_numpy(np.random.choice(np.arange(1, data.shape[0]), num_missing, replace=False)).long()
        data[missing_list] = 0.0
        has_value = Variable(torch.ones(data.shape[0], data.shape[1], 1))
        if use_gpu:
            has_value = has_value.cuda()
        has_value[missing_list] = 0.0
        data = torch.cat([has_value, data], 2)
        seq_len = data.shape[0]

        if teacher_forcing:
            batch_loss = model(data, ground_truth)
        else:
            data_list = []
            for j in range(seq_len):
                data_list.append(data[j:j+1])
            samples = model.sample(data_list)
            batch_loss = torch.mean((ground_truth - samples).pow(2))

        if train:
            optimizer.zero_grad()
            total_loss = batch_loss
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
        
        losses.append(batch_loss.data.cpu().numpy())

    return np.mean(losses)

def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)

def zeros(*shape):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)

# train and pretrain discriminator
def update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, \
                   states, actions, i_iter, dis_times, use_gpu, train = True):
    if use_gpu:
        exp_states, exp_actions, states, actions = exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()

    """update discriminator"""
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):
        g_o = discrim_net(Variable(states), Variable(actions))
        e_o = discrim_net(Variable(exp_states), Variable(exp_actions))
        
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        
        if train:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(g_o, Variable(zeros((g_o.shape[0], g_o.shape[1], 1)))) + \
                discrim_criterion(e_o, Variable(ones((e_o.shape[0], e_o.shape[1], 1))))
            discrim_loss.backward()
            optimizer_discrim.step()
    
    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times

# train policy network
def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, \
                  states_var, actions_var, i_iter, use_gpu):
    optimizer_policy.zero_grad()
    g_o = discrim_net(states_var, actions_var)
    policy_loss = discrim_criterion(g_o, Variable(ones((g_o.shape[0], g_o.shape[1], 1))))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 10)
    optimizer_policy.step()

# sample trajectories used in GAN training
def collect_samples_interpolate(policy_net, expert_data, use_gpu, i_iter, task, size=64, name="sampling_inter", draw=False, stats=False, num_missing=None):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    data = expert_data[exp_ind].clone()
    seq_len = data.shape[1]
    #print(data.shape, seq_len)
    if use_gpu:
        data = data.cuda()
    data = Variable(data.squeeze().transpose(0, 1))
    ground_truth = data.clone()
    
    if num_missing is None:
        num_missing = np.random.randint(seq_len * 4 // 5, seq_len)
        #num_missing = np.random.randint(seq_len * 18 // 20, seq_len)
        #num_missing = 40
    missing_list = torch.from_numpy(np.random.choice(np.arange(1, seq_len), num_missing, replace=False)).long()
    sorted_missing_list, _ = torch.sort(missing_list)
    print("collect sample:", sorted_missing_list)
    data[missing_list] = 0.0
    has_value = Variable(torch.ones(seq_len, size, 1))
    if use_gpu:
        has_value = has_value.cuda()
    has_value[missing_list] = 0.0
    data = torch.cat([has_value, data], 2)
    data_list = []
    for i in range(seq_len):
        data_list.append(data[i:i+1])
    samples = policy_net.sample(data_list)
    
    states = samples[:-1, :, :]
    actions = samples[1:, :, :]
    exp_states = ground_truth[:-1, :, :]
    exp_actions = ground_truth[1:, :, :]

    mod_stats = draw_and_stats(samples.data, name + '_' + str(num_missing), i_iter, task, draw=draw, compute_stats=stats, missing_list=missing_list)
    exp_stats = draw_and_stats(ground_truth.data, name + '_expert' + '_' + str(num_missing), i_iter, task, draw=draw, compute_stats=stats, missing_list=missing_list)
    
    return exp_states.data, exp_actions.data, ground_truth.data, states, actions, samples.data, mod_stats, exp_stats

# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret

def ave_player_distance(states):
    # states: numpy (seq_lenth, batch, 10)
    count = 0
    ret = np.zeros(states.shape)
    for i in range(5):
        for j in range(i+1, 5):
            ret[:, :, count] = np.sqrt(np.square(states[:, :, 2 * i] - states[:, :, 2 * j]) + \
                                       np.square(states[:, :, 2 * i + 1] - states[:, :, 2 * j + 1]))
            count += 1
    return ret

# draw and compute statistics
def draw_and_stats(model_states, name, i_iter, task, compute_stats=True, draw=True, missing_list=None):
    stats = {}
    if compute_stats:
        model_actions = model_states[1:, :, :] - model_states[:-1, :, :]
            
        val_data = model_states.cpu().numpy()
        val_actions = model_actions.cpu().numpy()
    
        step_size = np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2]))
        change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
        stats['ave_change_step_size'] = np.mean(change_of_step_size)
        val_seqlength = np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis = 0)
        stats['ave_length'] = np.mean(val_seqlength)  ## when sum along axis 0, axis 1 becomes axis 0
        stats['ave_out_of_bound'] = np.mean((val_data < -0.51) + (val_data > 0.51))
        # stats['ave_player_distance'] = np.mean(ave_player_distance(val_data))
        # stats['diff_max_min'] = np.mean(np.max(val_seqlength, axis=1) - np.min(val_seqlength, axis=1))
    
    if draw:
        print("Drawing")
        draw_data = model_states.cpu().numpy()[:, 0, :] 
        draw_data = unnormalize(draw_data, task)
        colormap = ['b', 'r', 'g', 'm', 'y']
        plot_sequence(draw_data, task, colormap=colormap, \
                      save_name="imgs/{}_{}".format(name, i_iter), missing_list=missing_list)

    return stats

def unnormalize(x, task):
    dim = x.shape[-1]
    
    if task == 'basketball':
        NORMALIZE = [94, 50] * int(dim / 2)
        SHIFT = [25] * dim
        return np.multiply(x, NORMALIZE) + SHIFT
    else:
        NORMALIZE = [128, 128] * int(dim / 2)
        SHIFT = [1] * dim
        return np.multiply(x + SHIFT, NORMALIZE)

def _set_figax(task):
    fig = plt.figure(figsize=(5,5))
    
    if task == 'basketball':
        img = plt.imread('data/court.png')
        img = resize(img,(500,940,3))
        ax = fig.add_subplot(111)
        ax.imshow(img)
    
        # show just the left half-court
        ax.set_xlim([-50,550])
        ax.set_ylim([-50,550])
        
    else:
        img = plt.imread('data/world.jpg')
        img = resize(img,(256,256,3))
    
        ax = fig.add_subplot(111)
        ax.imshow(img)
    
        ax.set_xlim([-50,300])
        ax.set_ylim([-50,300])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax

def plot_sequence(seq, task, colormap, save_name='', missing_list=None):
    n_players = int(len(seq[0])/2)

    while len(colormap) < n_players:
        colormap += 'b'

    fig, ax = _set_figax(task)
    if task == 'basketball':
        SCALE = 10
    else:
        SCALE = 1

    for k in range(n_players):
        x = seq[:,(2*k)]
        y = seq[:,(2*k+1)]
        color = colormap[k]
        ax.plot(SCALE*x, SCALE*y, color=color, linewidth=3, alpha=0.7)
        ax.plot(SCALE*x, SCALE*y, 'o', color=color, markersize=8, alpha=0.5)

    # starting positions
    x = seq[0,::2]
    y = seq[0,1::2]
    ax.plot(SCALE*x, SCALE*y, 'o', color='black', markersize=12)

    if missing_list is not None:
        missing_list = missing_list.numpy()
        for i in range(seq.shape[0]):
            if i not in missing_list:
                x = seq[i,::2]
                y = seq[i,1::2]
                ax.plot(SCALE*x, SCALE*y, 'o', color='black', markersize=8)

    plt.tight_layout(pad=0)

    if len(save_name) > 0:
        plt.savefig(save_name+'.png')
    else:
        plt.show()
    
    plt.close(fig)