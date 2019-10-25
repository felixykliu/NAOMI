# NAOMI

Code for NeurIPS 2019 paper titled [NAOMI: Non-Autoregressive Multiresolution Sequence Imputation](https://arxiv.org/abs/1901.10946)

Code is written with PyTorch v0.4.1 (Python 3.6.5). Billiards data can be downloaded [here](https://drive.google.com/open?id=17Ov4nwshLbn13w8qLuH8LNvzXzMTcjJt), basketball data is available from [STATS](https://www.stats.com/data-science/).

## To train the model:

First open visdom, then adjust hyperparameters in `train_model.sh` and run the shell file.

## Detailed explanations of hyperparameters:

•	`--model`: “NAOMI” or “SingleRes”

•	`--task`: “basketball” or “billiard”

•	`--y_dim`: 10 for basketball and 2 for billiard

•	`--rnn_dim` and `--n_layers`: gru cell size for all models, including forward and backward rnns

•	`--dec1_dim` to `--dec16_dim`: For NAOMI, these values correspond to dimensions of different decoders. For SingleRes, only dec1_dim is used for decoder.

•	`--pre_start_lr`: initial learning rate for supervised pretrain

•	`--pretrain`: supervised pretrain epochs

•	`--highest`: largest stepsize for NAOMI decoders, should be 2^n

•	`--discrim_rnn_dim` and `--discrim_layers`: discriminator rnn size

•	`--policy_learning_rate`: learning rate for generator in adversarial training

•	`--discrim_learning_rate`: learning rate for discriminator in adversarial training

•	`--pretrain_disc_iter`: number of iterations to pretrain discriminator

•	`--max_iter_num`: number of adversarial training iterations


## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{liu2019naomi,
  title={NAOMI: Non-Autoregressive Multiresolution Sequence Imputation},
  author={Liu, Yukai and Yu, Rose and Zheng, Stephan and Zhan, Eric and Yue, Yisong},
  booktitle={Advances in Neural Information Processing Systems(NeurIPS '19)},
  year={2019}
}
```
