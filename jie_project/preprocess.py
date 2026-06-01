import argparse

import torch
from torch.utils.data import DataLoader

from components.data_provider import Dataset_Preprocess_Building_Single, Dataset_Preprocess_Building_Multi
from components.models.preprocess_llama_model import PreprocessLlamaModel
from config import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMETimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default=config.llm_ckp_dir, help='llm checkpoints dir')
    parser.add_argument('--root_path', type=str, default='./dataset/building/', help='root path of the data file')
    parser.add_argument('--dataset', type=str, default='Hog_assembly_Colette.csv',
                        help='dataset to preprocess, options:[ETTh1, ETTh2, ETTm1, ETTm2, electricity, weather, traffic, electricity]')

    parser.add_argument('--seq_len', type=int, default=config.seq_len, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=config.label_len, help='label length')
    parser.add_argument('--pred_len', type=int, default=config.pred_len, help='pred length')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='pred length')
    args = parser.parse_args()
    print(args.dataset)
    
    model = PreprocessLlamaModel(args)

    seq_len = args.seq_len
    label_len = args.label_len
    pred_len = args.pred_len

    data_set = Dataset_Preprocess_Building_Single(
        root_path=args.root_path,
        data_path=args.dataset,
        size=[seq_len, label_len, pred_len])

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if isinstance(data_set, Dataset_Preprocess_Building_Single):
        print("use univariate statistic")
    elif isinstance(data_set, Dataset_Preprocess_Building_Multi):
        print("use multivarible statistic")

    from tqdm import tqdm
    print(len(data_set.data_stamp))
    save_dir_path = 'dataset/building/'
    output_list = []
    for idx, data in tqdm(enumerate(data_loader)):
        output = model(data)
        output_list.append(output.detach().cpu())
    result = torch.cat(output_list, dim=0)
    print(result.shape)
    print(result)
    torch.save(result, save_dir_path + f'/{args.dataset.strip(".csv")}.pt')
