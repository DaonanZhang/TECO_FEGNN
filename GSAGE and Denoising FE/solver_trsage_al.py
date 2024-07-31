import os, glob, inspect, time, math, torch, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import Dataloader_trsage_al as dl
from model_trsage_al import *
import torch.optim as optim
from torch.nn import functional as F
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D

import myconfig_trsage_al as myconfig
from datetime import datetime
import json
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
import random
import tqdm

import sys
from gauxlearn.optim import MetaOptimizer
import support_functions
from torch.autograd import gradcheck, grad

from torchviz import make_dot

class MaskedMAELoss(nn.Module):
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, pred, target):
        mask_value = -float('inf')
        mask = target != mask_value

        masked_pred = pred[mask]
        masked_target = target[mask]

        if masked_target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        L1_Loss = nn.L1Loss(reduction='mean')(masked_pred, masked_target)
        return L1_Loss



def map_param_to_block(shared_params):
    param_to_block = {}

    for i, (name, param) in enumerate(shared_params):
        if 'fc' in name:
            param_to_block[i] = 0
        elif 'd_' in name:
            param_to_block[i] = 1
        elif 's_' in name:
            param_to_block[i] = 2
        elif 'gsage' in name:
            param_to_block[i] = 3

    module_num = 4

    return param_to_block, module_num


class hypermodel(nn.Module):
    def __init__(self, task_num, module_num, param_to_block):

        super(hypermodel, self).__init__()
        self.task_num = task_num
        self.module_num = module_num
        self.param_to_block = param_to_block
        # task_num = settings['aux_task_num'] = 1, module_num = 4
        self.modularized_lr = nn.Parameter(torch.ones(task_num, module_num))
        self.nonlinear = nn.ReLU()
        self.scale_factor = 1.0

    def forward(self, loss_vector, shared_params, whether_single=1, train_lr=1.0):

        if whether_single == 1:
            grads = torch.autograd.grad(loss_vector[0], shared_params, create_graph=True)
            if self.nonlinear is not None:
                grads = tuple(self.nonlinear(self.modularized_lr[0][self.param_to_block[m]]) * g * train_lr for m, g in
                              enumerate(grads))
            else:
                grads = tuple(
                    self.modularized_lr[0][self.param_to_block[m]] * g * train_lr for m, g in enumerate(grads))
            return grads
        else:
            # always in this case!
            # main target loss and grad


            # for idx, p in enumerate(shared_params):
            #     print(f"Param {idx}: {p.name}, Requires grad: {p.requires_grad}")
                
            
            grads = torch.autograd.grad(loss_vector[0], shared_params, create_graph=True, allow_unused=True)
            loss_num = len(loss_vector)

            for task_id in range(1, loss_num):
                try:
                    loss_value = loss_vector[task_id].item()
                    zero_grads = [torch.zeros_like(param) for param in shared_params]
                    if loss_value == 0.0:
                        aux_grads = zero_grads
                    else:
                        aux_grads = torch.autograd.grad(loss_vector[task_id], shared_params, create_graph=True)
                    # aux_grads = torch.autograd.grad(loss_vector[task_id], shared_params, create_graph=True)
                    if self.nonlinear is not None:
                        # always in this case!
                        # tupel with len: len(grads, aux_grads)
                        # g:grads, g_aux:aux_grads, m:index in zip()--len(grads)
                        grads = tuple((g + self.scale_factor * self.nonlinear(
                            self.modularized_lr[task_id - 1][self.param_to_block[m]]) * g_aux) * train_lr for
                                      m, (g, g_aux) in enumerate(zip(grads, aux_grads)))
                    else:
                        grads = tuple((g + self.scale_factor * self.modularized_lr[task_id - 1][
                            self.param_to_block[m]] * g_aux) * train_lr for m, (g, g_aux) in
                                      enumerate(zip(grads, aux_grads)))
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'{[loss_vector[id] for id in range(1, loss_num)]}')
                    sys.exit()
            return grads


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
    
    dataset_train2 = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr2 = torch.utils.data.DataLoader(dataset_train2, batch_size=settings['batch'], shuffle=True,
                                                 collate_fn=dl.collate_fn, num_workers=4, prefetch_factor=32,
                                                 drop_last=True)

    aux_loader_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='aux')
    aux_loader_tr = torch.utils.data.DataLoader(aux_loader_train, batch_size=settings['batch'], shuffle=True,
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
    
    # # -----------------MAOAL model-----------------
    shared_parameter = [param for name, param in model.named_parameters() if 'task' not in name]
    shared_parameter1 = [(name, param) for name, param in model.named_parameters() if 'task' not in name]

    for name, param in shared_parameter1:
        print(f'name: {name}, param: {param.shape}')

    param_to_block, module_num = map_param_to_block(shared_parameter1)

    print(f'{param_to_block}')

    # whethersingle always = 0
    modular = hypermodel(settings['aux_task_num'], module_num, param_to_block).to(device)

    m_optimizer = optim.SGD(modular.parameters(), lr=settings['hyper_lr'], momentum=0.9,
                            weight_decay=settings['hyper_decay'])

    meta_optimizer = MetaOptimizer(meta_optimizer=m_optimizer, hpo_lr=1.0, truncate_iter=3, max_grad_norm=10)

    
    # loss_func = torch.nn.L1Loss()
    loss_func = model.loss_func

    aux_loss_func = MaskedMAELoss()

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

    
    aux_iter_loss = [torch.tensor(0., device=device) for _ in range(settings['aux_task_num'])]
    aux_mini_loss = [torch.tensor(0., device=device) for _ in range(settings['aux_task_num'])]
    
    data_iter = iter(dataloader_tr)
    data_iter2 = iter(dataloader_tr2)
    aux_iter = iter(aux_loader_tr)

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
                        model_output, target_head, _, _ = model(batch)
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

            model_output, target_head, aux_outputs_b, aux_targets_b = model(batch)

            elbo, outputs_b, targets_b = loss_func(model_output, target_head)

            # batch_loss = loss_func(outputs_b, targets_b)
            batch_loss = elbo
            batch_loss /= settings['accumulation_steps']
            inter_loss += batch_loss.item()
            mini_loss += batch_loss.item()
            # batch_loss.backward()


            # ---------------- TEST -----------------
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Parameter {name} received gradient: {param.grad}")
                else:
                    print(f"Parameter {name} did not receive gradient.")

        
            # -----------------aux_loss-----------------
            aux_batch_loss_list = []
            for aux_output, aux_target in zip(aux_outputs_b, aux_targets_b):
                aux_loss = aux_loss_func(aux_output, aux_target)
                aux_batch_loss_list.append(
                    (aux_loss * settings['hyper_aux_loss_weight'] / settings['accumulation_steps']))
                
            for i in range(0, settings['aux_task_num']):
                aux_mini_loss[i] += aux_batch_loss_list[i]
                aux_iter_loss[i] += aux_batch_loss_list[i]

            # # -------------Backward in the minibatch with AUX_loss-------------
            loss_list = [batch_loss] + aux_batch_loss_list

            common_grads = modular(loss_list, shared_parameter, whether_single=0) 
            
            loss_vec = torch.stack(loss_list)
            total_loss = torch.sum(loss_vec)
            total_loss.backward()

             
            for p, g in zip(shared_parameter, common_grads):
                if (iter_counter) % settings['accumulation_steps'] == 0:
                    p.grad = g
                else:
                    p.grad += g
                    
            del common_grads

            
            # --------------optimizer.step-------------------
            if (iter_counter + 1) % settings['accumulation_steps'] == 0:

                # -------------optimizer.step-------------
                optimizer.step()
                optimizer.zero_grad()

                # -----------------log-----------------
                t_train_iter_end = time.time()
                print(
                    f'\tIter {real_iter} - Loss: {mini_loss} Aux_loss:{[loss for loss in aux_mini_loss]} - real_iter_time: {t_train_iter_end - t_train_iter_start}',
                    end="\r", flush=True)

                # -----------------reset-----------------
                mini_loss = torch.tensor(0., device=device)
                aux_mini_loss = [torch.tensor(0., device=device) for _ in range(settings['aux_task_num'])]
                t_train_iter_start = t_train_iter_end


            # -----------------optimize the Hyper Parameters-----------------
            if (real_iter + 1) % settings['hyper_interval'] == 0 and real_iter > settings['hyper_pre']:
    
                try:
                    aux_batch = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(aux_loader_tr)
                    aux_batch = next(aux_iter)
    
                try:
                    train_batch2 = next(data_iter2)
                except StopIteration:
                    data_iter2 = iter(dataloader_tr2)
                    train_batch2 = next(data_iter2)
    
                # ----------------------------meta_loss----------------------------
    
                meta_outputs_b, meta_targets_b, _, _ = model(aux_batch)
    
                meta_primary_loss, _ , _ = loss_func(meta_outputs_b, meta_targets_b)
                meta_primary_loss /= settings['accumulation_steps']
                meta_total_loss = meta_primary_loss
    
                # ----------------------------train_loss2----------------------------
                outputs_b, targets_b, aux_outputs_b, aux_targets_b = model(train_batch2)
    
                primary_loss, _, _ = loss_func(outputs_b, targets_b)
                primary_loss /= settings['accumulation_steps']
    
                # # -----------------aux_loss-----------------
                aux_loss_list = []
                for aux_output, aux_target in zip(aux_outputs_b, aux_targets_b):
                    aux_loss = aux_loss_func(aux_output, aux_target)
                    aux_loss_list.append(aux_loss * settings['hyper_aux_loss_weight'] / settings['accumulation_steps'])
    
                loss_list = [primary_loss] + aux_loss_list
                train_common_grads = modular(loss_list, shared_parameter, whether_single=0, train_lr=1.0)
    
                meta_optimizer.step(val_loss=meta_total_loss, train_grads=train_common_grads,
                                    aux_params=list(modular.parameters()), shared_parameters=shared_parameter)
    
                # -----------------log-----------------
                print(f'\tIter {real_iter} - Meta_Loss: {meta_total_loss.item()} - Main_loss: {primary_loss.item()}',
                      end="\r", flush=True)
    
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

    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)

    model.load_state_dict(torch.load(coffer_dir + "best_params"))

    # build dataloader
    rtn_mae_list = []
    rtn_rsq_list = []

    loss_func = model.loss_func

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

                model_output, target_head, _, _ = model(batch)

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