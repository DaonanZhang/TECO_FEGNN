import pickle

import torch
from matplotlib import pyplot as plt

import Dataloader_idw_baseline as dl
import myconfig_idw_baseline as myconfig
import solver
import support_functions

import json
import torch
import os
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats

def main(hold_out, seed):
    settings = {
        'agent_id': '00000',
        'agent_dir': './logs',
        'origin_path': './Dataset_res250_reg4c/',
        'debug': False,
        'bp': False,

        'batch': 32,
        'accumulation_steps': 1,
        'epoch': 1000,
        'test_batch': 64,
        'nn_lr': 1e-5,
        'es_mindelta': 0.5,
        'es_endure': 10,
        'model': 'PEGNN',
        'lowest_rank': 1,
    }

    job_id = f"{hold_out}_{seed}"
    
    settings['fold'] = 4
    settings['seed'] = seed
    settings['holdout'] = hold_out

    support_functions.seed_everything(settings['seed'])
    
    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']
    
    coffer_dir = ''
    dirs = os.listdir(myconfig.coffer_path)

    dirs.sort()
    for dir in dirs:
        if job_id in dir:
            coffer_dir = myconfig.coffer_path + dir + f'/{fold}/'
            os.mkdir(coffer_dir)

    # Get device setting
    if not torch.cuda.is_available(): 
        device = torch.device("cpu")
        ngpu = 0
        print(f'Working on CPU')
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            print(f'Working on multi-GPU {device_list}')
        else:
            print(f'Working on single-GPU')
            
    with open(settings['origin_path'] + f"Folds_Info/norm_{settings['fold']}_{settings['holdout']}.json", 'r') as f:
        dic_op_minmax = json.load(f)
   

    for mask_distance in [0, 10, 50]:
        dataset_eval = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='eval')
        dataloader_ev = torch.utils.data.DataLoader(dataset_eval, batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=2, prefetch_factor=4, drop_last=True)


        output_list = []
        target_list = []

        len_eval = len(dataloader_ev)
        print(f"length of the eval: {len_eval}")
       
        for x_b, c_b, y_b, input_lengths in dataloader_ev:
            
            x_b, c_b, y_b, input_lengths = x_b.to(device), c_b.to(device), y_b.to(device), input_lengths.to(device)

            
            outputs_b = x_b[:,0,0]
            
            targets_b = y_b[:,0,0]

            # print(f' features_value {outputs_b} answers_value {targets_b}')
            
            output_list.append(outputs_b.detach().cpu())
            target_list.append(targets_b.detach().cpu())
                    
        output = torch.cat(output_list).squeeze()
        target = torch.cat(target_list).squeeze()

        # -----------------restore result-----------------
        min_val = dic_op_minmax['mcpm10'][0]
        max_val = dic_op_minmax['mcpm10'][1]
        test_means_origin = output * (max_val - min_val) + min_val
        test_y_origin = target * (max_val - min_val) + min_val


        # Mean Absolute Error
        # mse = mean_squared_error(test_y_origin, test_means_origin, squared=False)
        mae = mean_absolute_error(test_y_origin, test_means_origin)
        r_squared = stats.pearsonr(test_y_origin, test_means_origin)

        # rtn_mae_list.append(float(mae))
        # rtn_rsq_list.append(float(r_squared[0]))

        print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MAE: {str(mae)}\n\t\t--------\n')
        print(f'\t\t--------\n\t\tDiffer: {test_means_origin.max() - test_means_origin.min()}, count: {test_y_origin.size(0)}\n\t\t--------\n')

        title = f'Fold{fold}_holdout{holdout}_Md{mask_distance}: MAE {round(mae, 2)} R2 {round(r_squared[0], 2)}'
        
        support_functions.save_square_img(
            contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
            xlabel='targets_ex', ylabel='output_ex', 
            savename=os.path.join(coffer_dir, f'result_{mask_distance}'),
            title=title
        )
        targets_ex = test_y_origin.unsqueeze(1)
        output_ex = test_means_origin.unsqueeze(1)
        diff_ex = targets_ex - output_ex
        
        pd_out = pd.DataFrame(
            torch.cat(
                (targets_ex, output_ex, diff_ex), 1
            ).numpy()
        )
        pd_out.columns = ['Target', 'Output', 'Diff']
        
        pd_out.to_csv(os.path.join(coffer_dir, f'result_{mask_distance}.csv'), index=False)


def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


if __name__ == "__main__":
    hold_out_list = [0,1,2,3]
    seed_list = [1,2,3]  

    for hold_out_i in hold_out_list:
        for seed_i in seed_list:
            
            job_id = f"{hold_out_i}_{seed_i}"
            
            coffer_slot = myconfig.coffer_path + str(job_id) + '/'
            make_dir(coffer_slot)
            
            main(hold_out_i, seed_i)

