import argparse
import random

import numpy as np
import torch
from components.forecast.long_term_forecasting import LongTermForecast

from config import config
from utils.print_args import print_args
from utils.save_result import record_runtime_metrics


if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SMETimes')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, '
                             'in_context_forecasting]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='SMETimes_Llama',
                        help='model name, options: [SMETimes_Llama]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default=config.data, help='dataset type')
    parser.add_argument('--root_path', type=str, default=config.root_path, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=config.data_path, help='data file')
    parser.add_argument('--test_data_path', type=str, default=config.test_data_path, help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default=config.checkpoints, help='location of model checkpoints')
    parser.add_argument('--drop_last',  action='store_true', default=False, help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=config.seq_len, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=config.label_len, help='label length')
    parser.add_argument('--pred_len', type=int, default=config.pred_len, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=config.seq_len, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=config.label_len, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=config.pred_len, help='test pred len')
    parser.add_argument('--inverse', action='store_true', default=False, help='save test outputs in original target scale')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default=config.llm_ckp_dir, help='llm checkpoints dir')
    parser.add_argument('--mlp_hidden_dim', type=int, default=config.mlp_hidden_dim, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=config.mlp_hidden_layers, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')
    parser.add_argument('--num_experts', type=int, default=1, help='num experts')
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='lambda reg')
    parser.add_argument('--in_channel', type=int, default=7, help='encoder input size')
    parser.add_argument('--out_channel', type=int, default=1, help='encoder output size')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=True)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=True)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--visualize', action='store_true', help='visualize', default=True)
    args = parser.parse_args()

    print_args(args)
    
    Exp = LongTermForecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.des, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            inference_profile = exp.test(setting)
            record_runtime_metrics(args, setting, getattr(exp, "training_profile", {}), inference_profile)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.des, ii)
        exp = Exp(args) 
        inference_profile = exp.test(setting, test=1)
        record_runtime_metrics(args, setting, {}, inference_profile)
        torch.cuda.empty_cache()
