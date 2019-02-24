import argparse
import os
import math
import sys
import pickle
import time
import numpy as np
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import *
from torch.autograd import Variable
from torch import nn
import torch
import torch.utils
import torch.utils.data

from helpers import *
import visdom

Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

def printlog(line):
    print(line)
    with open(save_path+'log.txt', 'a') as file:
        file.write(line+'\n')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True, help='NAOMI, SingleRes')
parser.add_argument('--task', type=str, required=True, help='basketball, billiard')
parser.add_argument('--y_dim', type=int, required=True)
parser.add_argument('--rnn_dim', type=int, required=True)
parser.add_argument('--dec1_dim', type=int, required=True)
parser.add_argument('--dec2_dim', type=int, required=True)
parser.add_argument('--dec4_dim', type=int, required=True)
parser.add_argument('--dec8_dim', type=int, required=True)
parser.add_argument('--dec16_dim', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=False, default=2)
parser.add_argument('--seed', type=int, required=False, default=123)
parser.add_argument('--clip', type=int, required=True, help='gradient clipping')
parser.add_argument('--pre_start_lr', type=float, required=True, help='pretrain starting learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=64)
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--pretrain', type=int, required=False, default=50, help='num epochs to use supervised learning to pretrain')
parser.add_argument('--highest', type=int, required=False, default=1, help='highest resolution in terms of step size in NAOMI')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')

parser.add_argument('--discrim_rnn_dim', type=int, required=True)
parser.add_argument('--discrim_layers', type=int, required=True, default=2)
parser.add_argument('--policy_learning_rate', type=float, default=1e-6, help='policy network learning rate for GAN training')
parser.add_argument('--discrim_learning_rate', type=float, default=1e-3, help='discriminator learning rate for GAN training')
parser.add_argument('--max_iter_num', type=int, default=60000, help='maximal number of main iterations (default: 60000)')
parser.add_argument('--log_interval', type=int, default=1, help='interval between training status logs (default: 1)')
parser.add_argument('--draw_interval', type=int, default=200, help='interval between drawing and more detailed information (default: 50)')
parser.add_argument('--pretrain_disc_iter', type=int, default=2000, help="pretrain discriminator iteration (default: 2000)")
parser.add_argument('--save_model_interval', type=int, default=50, help="interval between saving model (default: 50)")

args = parser.parse_args()

if not torch.cuda.is_available():
    args.cuda = False
    
# model parameters
params = {
    'task' : args.task,
    'batch' : args.batch_size,
    'y_dim' : args.y_dim,
    'rnn_dim' : args.rnn_dim,
    'dec1_dim' : args.dec1_dim,
    'dec2_dim' : args.dec2_dim,
    'dec4_dim' : args.dec4_dim,
    'dec8_dim' : args.dec8_dim,
    'dec16_dim' : args.dec16_dim,
    'n_layers' : args.n_layers,
    'discrim_rnn_dim' : args.discrim_rnn_dim,
    'discrim_num_layers' : args.discrim_layers,
    'cuda' : args.cuda,
    'highest' : args.highest,
}

# hyperparameters
pretrain_epochs = args.pretrain
clip = args.clip
start_lr = args.pre_start_lr
batch_size = args.batch_size
save_every = args.save_every

# manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

# build model
policy_net = eval(args.model)(params)
discrim_net = Discriminator(params).double()
if args.cuda:
    policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
params['total_params'] = num_trainable_params(policy_net)
print(params)

# create save path and saving parameters
save_path = 'saved/' + args.model + '_' + args.task + '_%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'model/')

# Data
if args.task == 'basketball':
    test_data = torch.Tensor(pickle.load(open('data/basketball_eval.p', 'rb'))).transpose(0, 1)[:, :-1, :]
    train_data = torch.Tensor(pickle.load(open('data/basketball_train.p', 'rb'))).transpose(0, 1)[:, :-1, :]
elif args.task == 'billiard':
    test_data = torch.Tensor(pickle.load(open('data/billiard_eval.p', 'rb'), encoding='latin1'))[:, :, :]
    train_data = torch.Tensor(pickle.load(open('data/billiard_train.p', 'rb'), encoding='latin1'))[:, :, :]
else:
    print('no such task')
    exit()
print(test_data.shape, train_data.shape)

# figures and statistics
if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')
vis = visdom.Visdom(env = args.model + args.task + str(args.trial))
win_pre_policy = None
win_pre_path_length = None
win_pre_out_of_bound = None
win_pre_step_change = None

############################################################################
##################       START SUPERVISED PRETRAIN        ##################
############################################################################

# pretrain
best_test_loss = 0
lr = start_lr
teacher_forcing = True
for e in range(pretrain_epochs):
    epoch = e+1
    print("Epoch: {}".format(epoch))

    # draw and stats 
    _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples_interpolate(policy_net, test_data, use_gpu, e, args.task, name='pretrain_inter', draw=True, stats=True)
            
    update = 'append' if epoch > 1 else None
    win_pre_path_length = vis.line(X = np.array([epoch]), \
        Y = np.column_stack((np.array([exp_stats['ave_length']]), np.array([mod_stats['ave_length']]))), \
        win = win_pre_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
    win_pre_out_of_bound = vis.line(X = np.array([epoch]), \
        Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), np.array([mod_stats['ave_out_of_bound']]))), \
        win = win_pre_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
    win_pre_step_change = vis.line(X = np.array([epoch]), \
        Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), np.array([mod_stats['ave_change_step_size']]))), \
        win = win_pre_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))

    # control learning rate
    if epoch == pretrain_epochs // 2:
        lr = lr / 10
        print(lr)
        
    if args.task == 'billiard' and epoch == pretrain_epochs * 2 // 3:
        teacher_forcing = False

    # train
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy_net.parameters()),
        lr=lr)

    start_time = time.time()

    train_loss = run_epoch(True, policy_net, train_data, clip, optimizer, teacher_forcing=teacher_forcing)
    printlog('Train:\t' + str(train_loss))

    test_loss = run_epoch(False, policy_net, test_data, clip, optimizer, teacher_forcing=teacher_forcing)
    printlog('Test:\t' + str(test_loss))

    epoch_time = time.time() - start_time
    printlog('Time:\t {:.3f}'.format(epoch_time))

    total_test_loss = test_loss
    
    update = 'append' if epoch > 1 else None
    win_pre_policy = vis.line(X = np.array([epoch]), Y = np.column_stack((np.array([test_loss]), np.array([train_loss]))), \
        win = win_pre_policy, update = update, opts=dict(legend=['out-of-sample loss', 'in-sample loss'], \
                                                         title="pretrain policy training curve"))

    # best model on test set
    if best_test_loss == 0 or total_test_loss < best_test_loss:    
        best_test_loss = total_test_loss
        filename = save_path+'model/policy_step'+str(args.highest)+'_state_dict_best_pretrain.pth'
        torch.save(policy_net.state_dict(), filename)
        printlog('Best model at epoch '+str(epoch))

    # periodically save model
    if epoch % save_every == 0:
        filename = save_path+'model/policy_step'+str(args.highest)+'_state_dict_'+str(epoch)+'.pth'
        torch.save(policy_net.state_dict(), filename)
        printlog('Saved model')
    
printlog('End of Pretrain, Best Test Loss: {:.4f}'.format(best_test_loss))

# billiard does not need adversarial training
if args.task == 'billiard':
    exit()

############################################################################
##################       START ADVERSARIAL TRAINING       ##################
############################################################################

# load the best pretrained policy
policy_state_dict = torch.load(save_path+'model/policy_step'+str(args.highest)+'_state_dict_best_pretrain.pth')
#policy_state_dict = torch.load(save_path+'model/policy_step'+str(args.highest)+'_training.pth')
policy_net.load_state_dict(policy_state_dict)
    
# optimizer
optimizer_policy = torch.optim.Adam(
    filter(lambda p: p.requires_grad, policy_net.parameters()),
    lr=args.policy_learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.discrim_learning_rate)
discrim_criterion = nn.BCELoss()
if use_gpu:
    discrim_criterion = discrim_criterion.cuda()

# stats
exp_p = []
win_exp_p = None
mod_p = []
win_mod_p = None
win_path_length = None
win_out_of_bound = None
win_step_change = None

# Pretrain Discriminator
for i in range(args.pretrain_disc_iter):
    exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples_interpolate(policy_net, train_data, use_gpu, i, args.task, name="pretraining", draw=False, stats=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data
    pre_mod_p, pre_exp_p = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, \
        exp_actions, model_states, model_actions, i, dis_times=3.0, use_gpu=use_gpu, train=True)

    print(i, 'exp: ', pre_exp_p, 'mod: ', pre_mod_p)

    if pre_mod_p < 0.3:
        break

# Save pretrained model
if args.pretrain_disc_iter > 250:
    torch.save(policy_net.state_dict(), save_path+'model/policy_step'+str(args.highest)+'_pretrained.pth')
    torch.save(discrim_net.state_dict(), save_path+'model/discrim_step'+str(args.highest)+'_pretrained.pth')
    
# GAN training
for i_iter in range(args.max_iter_num):
    ts0 = time.time()
    print("Collecting Data")
    exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples_interpolate(policy_net, train_data, use_gpu, i_iter, args.task, draw=False, stats=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data    
    
    # draw and stats
    if i_iter % args.draw_interval == 0:
        _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples_interpolate(policy_net, test_data, use_gpu, i_iter, args.task, draw=True, stats=True)
    
        # print(mod_stats)
        update = 'append' if i_iter > 0 else None
        win_path_length = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_length']]), np.array([mod_stats['ave_length']]))), \
            win = win_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
        win_out_of_bound = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), np.array([mod_stats['ave_out_of_bound']]))), \
            win = win_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
        win_step_change = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), np.array([mod_stats['ave_change_step_size']]))), \
            win = win_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))
        
    ts1 = time.time()

    t0 = time.time()
    # update discriminator
    mod_p_epoch, exp_p_epoch = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, \
                                              model_states, model_actions, i_iter, dis_times=3.0, use_gpu=use_gpu, train=True)
    exp_p.append(exp_p_epoch)
    mod_p.append(mod_p_epoch)
    
    # update policy network
    if i_iter > 3 and mod_p[-1] < 0.8:
        update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, model_states_var, model_actions_var, i_iter, use_gpu)
    t1 = time.time()

    if i_iter % args.log_interval == 0:
        print('{}\tT_sample {:.4f}\tT_update {:.4f}\texp_p {:.3f}\tmod_p {:.3f}'.format(
            i_iter, ts1-ts0, t1-t0, exp_p[-1], mod_p[-1]))
        
        update = 'append'
        if win_exp_p is None:
            update = None
        win_exp_p = vis.line(X = np.array([i_iter]), \
                             Y = np.column_stack((np.array([exp_p[-1]]), np.array([mod_p[-1]]))), \
                             win = win_exp_p, update = update, \
                             opts=dict(legend=['expert_prob', 'model_prob'], title="training curve probs"))

    if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
        torch.save(policy_net.state_dict(), save_path+'model/policy_step'+str(args.highest)+'_training.pth')
        torch.save(discrim_net.state_dict(), save_path+'model/discrim_step'+str(args.highest)+'_training.pth')
