import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import Dataloader_trsage as dl
from model_trsage import *
import torch.optim as optim
from torch.nn import functional as F
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D
import myconfig_trsage as myconfig
from datetime import datetime
import json
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
import random
import tqdm

import sys

import support_functions


# where is the auxilary task and the loss from there?
def bmc_loss(pred, target, noise_var, device):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))  # contrastive-like loss
    # loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable
    return loss


# lack of the file: support_functions.py !!

def training(settings, job_id):
    support_functions.seed_everything(settings['seed'])

    fold = settings['fold']
    holdout = settings['holdout']
    lowest_rank = settings['lowest_rank']

    coffer_slot = settings['coffer_slot'] + f'{fold}/'
    support_functions.make_dir(coffer_slot)

    # print sweep settings
    print(json.dumps(settings, indent=2, default=str))

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

    # build dataloader
    dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['batch'], shuffle=True,
                                                collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=32,
                                                drop_last=True)

    test_dataloaders = []
    for mask_distance in [10]:
        test_dataloaders.append(
            torch.utils.data.DataLoader(
                dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='test'),
                batch_size=settings['batch'], shuffle=False, collate_fn=dl.collate_fn, num_workers=4,
                prefetch_factor=32, drop_last=True
            )
        )

    # build model
    model = INTP_Model(settings=settings, device=device).to(device)
    model = model.float()

    # loss_func = torch.nn.L1Loss()
    loss_func = model.loss_func

    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings['nn_lr'])

    # set training loop
    epochs = settings['epoch']
    batch_size = settings['batch']
    print("\nTraining to %d epochs (%d of mini batch size)" % (epochs, batch_size))

    # fire training loop
    start_time = time.time()
    list_total = []
    list_err = []

    best_err = float('inf')

    # breakpoint:TODO:
    # best_err =
    # breaK_epoch =
    # load param =

    es_counter = 0

    iter_counter = 0
    inter_loss = 0
    mini_loss = 0
    epoch_counter = 0

    data_iter = iter(dataloader_tr)

    t_train_iter_start = time.time()

    steps = len(data_iter) / settings['accumulation_steps']
    print(f'Each epoch #real_iter: {steps}')

    print("working on training loop")

    while True:

        try:
            batch = next(data_iter)

        except StopIteration:
            model.eval()
            output_list = []
            target_list = []
            test_loss = 0

            with torch.no_grad():
                for dataloader_ex in test_dataloaders:
                    print(f'total test: {len(dataloader_ex)}')

                    for batch in dataloader_ex:
                        model_output, target_head = model(batch)
                        elbo, outputs_b, targets_b = loss_func(model_output, target_head)

                        output_list.append(outputs_b)
                        target_list.append(targets_b)

            print('Test Done')
            output = torch.cat(output_list)
            target = torch.cat(target_list)

            test_loss = torch.nn.L1Loss(reduction='sum')(output, target).item()

            output = output.squeeze().detach().cpu()
            target = target.squeeze().detach().cpu()

            # -----------------restore result-----------------
            min_val = dic_op_minmax["mcpm10"][0]
            max_val = dic_op_minmax["mcpm10"][1]
            test_means_origin = output * (max_val - min_val) + min_val
            test_y_origin = target * (max_val - min_val) + min_val

            mae = mean_absolute_error(test_y_origin, test_means_origin)
            r_squared = stats.pearsonr(test_y_origin, test_means_origin)

            print(f'\t\t--------\n\t\tIter: {str(real_iter)}, inter_train_loss: {inter_loss}\n\t\t--------\n')
            print(
                f'\t\t--------\n\t\ttest_loss: {str(test_loss)}, last best test_loss: {str(best_err)}\n\t\t--------\n')
            print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MAE: {str(mae)}\n\t\t--------\n')

            if best_err - test_loss > settings['es_mindelta']:

                best_err = test_loss
                torch.save(model.state_dict(), coffer_slot + "best_params")

                support_functions.save_square_img(
                    contents=[test_y_origin.numpy(), test_means_origin.numpy()],
                    xlabel='targets_ex', ylabel='output_ex',
                    savename=os.path.join(coffer_slot, f'test_{epoch_counter}'),
                    title=f'Fold{fold}_holdout{holdout}_Md_all: MAE {round(mae, 2)} R2 {round(r_squared[0], 2)}'
                )
                es_counter = 0

            else:
                es_counter += 1
                print(f"INFO: Early stopping counter {es_counter} of {settings['es_endure']}")
                if es_counter >= settings['es_endure']:
                    print('INFO: Early stopping')
                    es_flag = 1
                    break

            list_err.append(float(test_loss))
            list_total.append(float(inter_loss))
            inter_loss = 0
            epoch_counter += 1

            print(f'Current epoch: {epoch_counter}')

            data_iter = iter(dataloader_tr)
            batch = next(data_iter)

            if epoch_counter > settings['epoch']:
                print('Finished Training')
                break

        finally:
            model.train()

            real_iter = iter_counter // settings['accumulation_steps'] + 1

            model_output, target_head = model(batch)

            elbo, outputs_b, targets_b = loss_func(model_output, target_head)

            # batch_loss = loss_func(outputs_b, targets_b)
            batch_loss = elbo

            batch_loss /= settings['accumulation_steps']

            inter_loss += batch_loss.item()
            mini_loss += batch_loss.item()
            # backward propagation
            batch_loss.backward()

            if (iter_counter + 1) % settings['accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                t_train_iter_end = time.time()
                print(
                    f'\tIter {real_iter} - Loss: {mini_loss} - real_iter_time: {t_train_iter_end - t_train_iter_start}',
                    end="\r", flush=True)
                mini_loss = 0
                t_train_iter_start = t_train_iter_end
            iter_counter += 1

    return list_total, list_err


# to evaluate the result of training
def evaluate(settings, job_id):
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
            break

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

    # build model
    model = INTP_Model(settings=settings, device=device).to(device)

    model = model.float()
    
    loss_func = model.loss_func

    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)

    model.load_state_dict(torch.load(coffer_dir + "best_params"))

    # build dataloader
    rtn_mae_list = []
    rtn_rsq_list = []


    for mask_distance in [10]:
        dataset_eval = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='eval')
        dataloader_ev = torch.utils.data.DataLoader(dataset_eval, batch_size=settings['batch'], shuffle=False,
                                                    collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=32,
                                                    drop_last=True)

        model.eval()

        output_list = []
        target_list = []

        with torch.no_grad():
            for batch in dataloader_ev:

                model_output, target_head = model(batch)

                elbo, outputs_b, targets_b = loss_func(model_output, target_head)

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
        mae = mean_absolute_error(test_y_origin, test_means_origin)
        r_squared = stats.pearsonr(test_y_origin, test_means_origin)

        rtn_mae_list.append(float(mae))
        rtn_rsq_list.append(float(r_squared[0]))

        print(f'\t\t--------\n\t\tr_squared: {str(r_squared[0])}, MAE: {str(mae)}\n\t\t--------\n')
        print(
            f'\t\t--------\n\t\tDiffer: {test_means_origin.max() - test_means_origin.min()}, count: {test_y_origin.size(0)}\n\t\t--------\n')

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

    return rtn_mae_list, rtn_rsq_list
