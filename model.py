import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

from model_utils import *

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total

class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.hidden_dim = params['discrim_rnn_dim']
        self.action_dim = params['y_dim']
        self.state_dim = params['y_dim']
        self.gpu = params['cuda']
        self.num_layers = params['discrim_num_layers'] 

        self.gru = nn.GRU(self.state_dim, self.hidden_dim, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, 1)
            
    def forward(self, x, a, h=None):  # x: seq * batch * 10, a: seq * batch * 10
        p, hidden = self.gru(x, h)   # p: seq * batch * 10
        p = torch.cat([p, a], 2)   # p: seq * batch * 20
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))    # prob: seq * batch * 1
        return prob

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

class NAOMI(nn.Module):

    def __init__(self, params):
        super(NAOMI, self).__init__()

        self.params = params
        self.task = params['task']
        self.stochastic = (self.task == 'basketball')
        self.y_dim = params['y_dim']
        self.rnn_dim = params['rnn_dim']
        self.dims = {}
        self.n_layers = params['n_layers']
        self.networks = {}
        self.highest = params['highest']
        self.batch_size = params['batch']

        self.gru = nn.GRU(self.y_dim, self.rnn_dim, self.n_layers)
        self.back_gru = nn.GRU(self.y_dim + 1, self.rnn_dim, self.n_layers)
        
        step = 1
        while step <= self.highest:
            l = str(step)
            self.dims[l] = params['dec' + l + '_dim']
            dim = self.dims[l]
            
            curr_level = {}
            curr_level['dec'] = nn.Sequential(
                nn.Linear(2 * self.rnn_dim, dim),
                nn.ReLU())
            curr_level['mean'] = nn.Linear(dim, self.y_dim)
            if self.stochastic:
                curr_level['std'] = nn.Sequential(
                    nn.Linear(dim, self.y_dim),
                    nn.Softplus())
            curr_level = nn.ModuleDict(curr_level)

            self.networks[l] = curr_level
            
            step = step * 2

        self.networks = nn.ModuleDict(self.networks)

    def forward(self, data, ground_truth):
        # data: seq_length * batch * 11
        # ground_truth: seq_length * batch * 10
        h = Variable(torch.zeros(self.n_layers, self.batch_size, self.rnn_dim))
        h_back = Variable(torch.zeros(self.n_layers, self.batch_size, self.rnn_dim))
        if self.params['cuda']:
            h, h_back = h.cuda(), h_back.cuda()
        
        loss = 0.0
        h_back_dict = {}
        count = 0
        
        for t in range(data.shape[0] - 1, 0, -1):
            h_back_dict[t+1] = h_back
            state_t = data[t]
            _, h_back = self.back_gru(state_t.unsqueeze(0), h_back)
            
        for t in range(data.shape[0]):
            state_t = ground_truth[t]
            _, h = self.gru(state_t.unsqueeze(0), h)
            count += 1
            for l, dim in self.dims.items():
                step_size = int(l)
                curr_level = self.networks[str(step_size)] 
                if t + 2 * step_size <= data.shape[0]:
                    next_t = ground_truth[t+step_size]
                    h_back = h_back_dict[t+2*step_size]
                    
                    dec_t = curr_level['dec'](torch.cat([h[-1], h_back[-1]], 1))
                    dec_mean_t = curr_level['mean'](dec_t)
                    
                    if self.stochastic:
                        dec_std_t = curr_level['std'](dec_t)
                        loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                    else:
                        loss += torch.sum((dec_mean_t - next_t).pow(2))

        return loss / count / data.shape[1]

    def sample(self, data_list):
        # data_list: seq_length * (1 * batch * 11)
        ret = []
        seq_len = len(data_list)
        h = Variable(torch.zeros(self.params['n_layers'], self.batch_size, self.rnn_dim))
        if self.params['cuda']:
            h = h.cuda()
        
        h_back_dict = {}
        h_back = Variable(torch.zeros(self.params['n_layers'], self.batch_size, self.rnn_dim))
        if self.params['cuda']:
            h_back = h_back.cuda()  
        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t+1] = h_back
            state_t = data_list[t]
            _, h_back = self.back_gru(state_t, h_back)
        
        curr_p = 0
        _, h = self.gru(data_list[curr_p][:, :, 1:], h)
        while curr_p < seq_len - 1:
            if data_list[curr_p + 1][0, 0, 0] == 1:
                curr_p += 1
                _, h = self.gru(data_list[curr_p][:, :, 1:], h)
            else:
                next_p = curr_p + 1
                while next_p < seq_len and data_list[next_p][0, 0, 0] == 0:
                    next_p += 1
                
                step_size = 1
                while curr_p + 2 * step_size <= next_p and step_size <= self.highest:
                    step_size *= 2
                step_size = step_size // 2
                
                self.interpolate(data_list, curr_p, h, h_back_dict, step_size)
        
        return torch.cat(data_list, dim=0)[:, :, 1:]

    def interpolate(self, data_list, curr_p, h, h_back_dict, step_size):
        #print("interpolating:", len(ret), step_size)
        h_back = h_back_dict[curr_p + 2 * step_size]
        curr_level = self.networks[str(step_size)]
        
        dec_t = curr_level['dec'](torch.cat([h[-1], h_back[-1]], 1))
        dec_mean_t = curr_level['mean'](dec_t)
        if self.stochastic:
            dec_std_t = curr_level['std'](dec_t)
            state_t = reparam_sample_gauss(dec_mean_t, dec_std_t)
        else:
            state_t = dec_mean_t
        
        added_state = state_t.unsqueeze(0)
        has_value = Variable(torch.ones(added_state.shape[0], added_state.shape[1], 1))
        if self.params['cuda']:
            has_value = has_value.cuda()
        added_state = torch.cat([has_value, added_state], 2)
        
        if step_size > 1:
            right = curr_p + step_size
            left = curr_p + step_size // 2
            h_back = h_back_dict[right+1]
            _, h_back = self.back_gru(added_state, h_back)
            h_back_dict[right] = h_back
            
            zeros = Variable(torch.zeros(added_state.shape[0], added_state.shape[1], self.y_dim + 1))
            if self.params['cuda']:
                zeros = zeros.cuda()
            for i in range(right-1, left-1, -1):
                _, h_back = self.back_gru(zeros, h_back)
                h_back_dict[i] = h_back
        
        data_list[curr_p + step_size] = added_state
    
class SingleRes(nn.Module):

    def __init__(self, params):
        super(SingleRes, self).__init__()

        self.params = params
        self.task = params['task']
        self.stochastic = (self.task == 'basketball')
        self.y_dim = params['y_dim']
        self.rnn_dim = params['rnn_dim']
        self.dec_dim = params['dec1_dim']
        self.n_layers = params['n_layers']
        self.networks = {}
        self.batch_size = params['batch']

        self.gru = nn.GRU(self.y_dim, self.rnn_dim, self.n_layers)
        self.back_gru = nn.GRU(self.y_dim + 1, self.rnn_dim, self.n_layers)
        self.dec = nn.Sequential(
                nn.Linear(2 * self.rnn_dim, self.dec_dim),
                nn.ReLU())
        self.mean = nn.Linear(self.dec_dim, self.y_dim)
        if self.stochastic:
            self.std = nn.Sequential(
                    nn.Linear(self.dec_dim, self.y_dim),
                    nn.Softplus())

    def forward(self, data, ground_truth):
        # data: seq_length * batch * 11
        # ground_truth: seq_length * batch * 10
        h = Variable(torch.zeros(self.n_layers, data.size(1), self.rnn_dim))
        h_back = Variable(torch.zeros(self.n_layers, data.size(1), self.rnn_dim))
        if self.params['cuda']:
            h, h_back = h.cuda(), h_back.cuda()
        
        loss = 0.0
        h_back_dict = {}
        count = 0
        for t in range(data.shape[0] - 1, 0, -1):
            h_back_dict[t+1] = h_back[-1]
            state_t = data[t]
            _, h_back = self.back_gru(state_t.unsqueeze(0), h_back)
            
        for t in range(data.shape[0] - 1):
            state_t = ground_truth[t]
            next_t = ground_truth[t+1]
            h_back = h_back_dict[t+2]
            
            _, h = self.gru(state_t.unsqueeze(0), h)
            dec_t = self.dec(torch.cat([h[-1], h_back], 1))
            dec_mean_t = self.mean(dec_t)

            if self.stochastic:
                dec_std_t = self.std(dec_t)
                loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
            else:
                loss += torch.sum((dec_mean_t - next_t).pow(2))
                
            count += 1

        return loss / count / data.shape[1]

    def sample(self, data_list):
        # data_list: seq_length * (1 * batch * 11)
        h = Variable(torch.zeros(self.n_layers, self.batch_size, self.rnn_dim))
        h_back = Variable(torch.zeros(self.n_layers, self.batch_size, self.rnn_dim))
        if self.params['cuda']:
            h, h_back = h.cuda(), h_back.cuda()

        seq_len = len(data_list)
        h_back_dict = {}
        count = 0
        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t+1] = h_back[-1]
            _, h_back = self.back_gru(data_list[t], h_back)

        for t in range(seq_len - 1):
            state_t = data_list[t][:, :, 1:]
            _, h = self.gru(state_t, h)
            if data_list[t+1][0, 0, 0] == 0:
                h_back = h_back_dict[t+2]
                dec_t = self.dec(torch.cat([h[-1], h_back], 1))
                dec_mean_t = self.mean(dec_t)
                
                if self.stochastic:
                    dec_std_t = self.std(dec_t)
                    state_t = reparam_sample_gauss(dec_mean_t, dec_std_t)
                else:
                    state_t = dec_mean_t
                    
                added_state = state_t.unsqueeze(0)
                has_value = Variable(torch.ones(added_state.shape[0], added_state.shape[1], 1))
                if self.params['cuda']:
                    has_value = has_value.cuda()
                added_state = torch.cat([has_value, added_state], 2)
                
                data_list[t + 1] = added_state
            
        return torch.cat(data_list, dim=0)[:, :, 1:]