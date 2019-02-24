#!/bin/bash

python train.py \
--trial 1 \
--model NAOMI \
--task basketball \
--y_dim 10 \
--rnn_dim 300 \
--dec1_dim 200 \
--dec2_dim 200 \
--dec4_dim 200 \
--dec8_dim 200 \
--dec16_dim 200 \
--n_layers 2 \
--clip 10 \
--pre_start_lr 1e-3 \
--batch_size 64 \
--pretrain 50 \
--highest 8 \
--discrim_rnn_dim 128 \
--discrim_layers 1 \
--policy_learning_rate 3e-6 \
--discrim_learning_rate 1e-3 \
--pretrain_disc_iter 2000 \
--max_iter_num 80000 \
--draw_interval 200 \
--cuda
