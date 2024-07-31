import torch
import os
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from osgeo import gdal
import concurrent.futures
import json


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.settings = settings
        self.mask_distance = mask_distance
        self.call_name = call_name
        self.origin_path = f"./Dataset_res250_reg4c/"

        with open(self.origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)
        with open(self.origin_path + f"Folds_Info/norm_{settings['fold']}_{settings['holdout']}.json", 'r') as f:
            self.dic_op_minmax = json.load(f)
        with open(self.origin_path + f"Folds_Info/divide_set_{settings['fold']}_{settings['holdout']}.info", 'rb') as f:
            divide_set = pickle.load(f)

        # load file list
        if call_name == 'train':
            call_scene_list = divide_set[0]
        elif call_name == 'test':
            call_scene_list = divide_set[1]
        elif call_name == 'eval':
            call_scene_list = divide_set[2]
        if settings['debug'] and len(call_scene_list) > 2000:
            call_scene_list = call_scene_list[:200]

        self.total_df_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
            for future in concurrent.futures.as_completed(futures):
                file_name, file_content = future.result()
                self.total_df_dict[file_name] = file_content

        self.call_list = []
        target_op_range = self.dataset_info["holdouts"][str(self.settings['holdout'])][self.call_name]
        for scene in call_scene_list:
            df = self.total_df_dict[scene]
            target_row_index_list = list(df[df['op'].isin(target_op_range)].index)
            target_row_index_list.sort()
            
            for index in target_row_index_list:
                self.call_list.append([scene, index])
                
        tail_index = (len(self.call_list) // 256) * 256
        self.call_list = self.call_list[:tail_index]

        print(f"Length of df dict: {len(list(self.total_df_dict.keys()))}")
        print(f"Length of call list: {len(self.call_list)}")


    def __len__(self):
        return len(self.call_list)

    def process_child(self, filename):
        df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
        # drop everything in bad quality
        df = df[df['Thing']>=self.dataset_info['lowest_rank']]
        # drop everything with non-whitelisted op
        op_whitelist = list(self.dataset_info["op_dic"].keys())
        for holdout in self.dataset_info["holdouts"].keys():
            op_whitelist = op_whitelist + self.dataset_info["holdouts"][holdout]["train"] + self.dataset_info["holdouts"][holdout]["test"] + self.dataset_info["holdouts"][holdout]["eval"]
        op_whitelist = list(set(op_whitelist))
        df = df[df['op'].isin(op_whitelist)]
        # normalize all values (coordinates will be normalized later)
        df = self.norm(d=df)
        return filename, df


    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d[d['op']==op].copy()
            if op in self.dataset_info["tgt_logical"]:
                op_norm = self.dataset_info["tgt_op"]
            else:
                op_norm = op
            d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])
            d_list.append(d_op)
        return pd.concat(d_list, axis=0, ignore_index=False).drop(columns=['Result'])

    def process_child(self, filename):
        df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
        # drop everything in bad quality
        df = df[df['Thing'] >= self.dataset_info['lowest_rank']]
        # drop everything with non-whitelisted op
        op_whitelist = list(self.dataset_info["op_dic"].keys())
        for holdout in self.dataset_info["holdouts"].keys():
            op_whitelist = op_whitelist + self.dataset_info["holdouts"][holdout]["train"] + \
                           self.dataset_info["holdouts"][holdout]["test"] + self.dataset_info["holdouts"][holdout][
                               "eval"]
        op_whitelist = list(set(op_whitelist))
        df = df[df['op'].isin(op_whitelist)]
        # normalize all values (coordinates will be normalized later)
        df = self.norm(d=df)
        return filename, df

    def distance_matrix(self, x0, y0, x1, y1):
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T
        d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
        d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
        # calculate hypotenuse
        return np.hypot(d0, d1)

    def idw_interpolation(self, x, y, values, xi, yi, p=2):
        dist = self.distance_matrix(x, y, xi, yi)
        # In IDW, weights are 1 / distance
        weights = 1.0 / (dist + 1e-12) ** p
        # Make weights sum to one
        weights /= weights.sum(axis=0)
        # Multiply the weights for each interpolated point by all observed Z-values
        return np.dot(weights.T, values)

    def norm_fcol(self, df):
        rtn = df.copy()
        # Norm other columns
        for col, (min_val, max_val) in self.dataset_info["eu_col"].items():
            rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
        for col, (min_val, max_val) in self.dataset_info["non_eu_col"].items():
            rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
        return rtn

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get call_item and scenario_story
        # load data item
        df = self.total_df_dict[self.call_list[idx][0]]
        target_row_index = self.call_list[idx][1]
        
        # keep target row
        target_row = pd.DataFrame(df.loc[target_row_index]).transpose()
        target_row = target_row.apply(pd.to_numeric, errors='ignore')
        df = df.drop(target_row_index)
        # clean unrelated labels
        keep_op_range = list(set(
            list(self.dataset_info["op_dic"].keys()) + 
            self.dataset_info["holdouts"][str(self.settings['holdout'])]["train"]
        ))
        df = df[df['op'].isin(keep_op_range)]

         # processing senario informations:
        #     - mask out all readings within 'mask_distance'
        if self.mask_distance == -1:
            this_mask = self.call_list[idx][2]
        else:
            this_mask = self.mask_distance
       
        df_filtered = df.loc[(abs(df['Longitude'] - target_row['Longitude'].values[0]) + abs(df['Latitude'] - target_row['Latitude'].values[0])) >= this_mask, :].copy()

        df_filtered.loc[df_filtered['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]
        target_row.loc[target_row['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]

        # _________________________________________Seperate frames_______________________________________________________
        df_known = df_filtered[df_filtered['op']==self.dataset_info["tgt_op"]].copy()
        df_auxil = df_filtered[df_filtered['op']!=self.dataset_info["tgt_op"]].copy()

        # _________________________________________Target features_______________________________________________________
        graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)
        
        coords = torch.from_numpy(graph_candidates[['Longitude', 'Latitude']].values).float()
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

        full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
        aggregated_df = full_df.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()
        features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
        possible_values = list(self.dataset_info["op_dic"].keys())
        possible_values.sort()

        for op in possible_values:
            aggregated_df_op = aggregated_df[aggregated_df['op']==op]
            interpolated_grid = torch.zeros((len(graph_candidates), 1))
            if len(aggregated_df_op) != 0:
                xi = graph_candidates['Longitude'].values
                yi = graph_candidates['Latitude'].values
                x = aggregated_df_op['Longitude'].values
                y = aggregated_df_op['Latitude'].values
                values = aggregated_df_op['Result_norm'].values
                interpolated_values = self.idw_interpolation(x, y, values, xi, yi)
                interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
            features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op]+1] = interpolated_grid
        
        conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
        features[:, -1] = torch.from_numpy(np.where(conditions, graph_candidates['Thing'] / self.dataset_info["non_eu_col"]["Thing"][1], (self.dataset_info['lowest_rank']-1)/self.dataset_info["non_eu_col"]["Thing"][1]))
        
        features = features.float()
        # use mask instead of idw for the target row
        features[0, 0:len(self.dataset_info["op_dic"])] = 0
        
        return features, coords, answers


# collate_fn: how samples are batched together
def collate_fn(examples):
    input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
    x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
    c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
    y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)
    return x_b, c_b, y_b, input_lenths