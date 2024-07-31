import json
import pickle
import concurrent.futures
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import support_functions


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.settings = settings
        self.mask_distance = mask_distance
        self.call_name = call_name

        # support_functions.seed_everything(settings['seed'])

        # load dataset info
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
            call_scene_list = call_scene_list[:2000]

        # do op filtering and normalization in the parallel fashion
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
            # if call_name == 'test' and len(target_row_index_list) >= 2:
            #     target_row_index_list = target_row_index_list[:2]
            for index in target_row_index_list:
                self.call_list.append([scene, index])
        tail_index = (len(self.call_list) // 256) * 256
        self.call_list = self.call_list[:tail_index]

        print(f"Length of df dict: {len(list(self.total_df_dict.keys()))}")
        print(f"Length of call list: {len(self.call_list)}")

    def __len__(self):
        return len(self.call_list)

    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d[d['op'] == op].copy()
            if op in self.dataset_info["tgt_logical"]:
                op_norm = self.dataset_info["tgt_op"]
            else:
                op_norm = op
            d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (
                    self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])
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

        # load data item
        df = self.total_df_dict[self.call_list[idx][0]]
        target_row_index = self.call_list[idx][1]

        # keep target row
        target_row = pd.DataFrame(df.loc[target_row_index]).transpose()
        target_row = target_row.apply(pd.to_numeric, errors='coerce')
        df = df.drop(target_row_index)
        # clean unrelated labels
        keep_op_range = list(set(
            list(self.dataset_info["op_dic"].keys()) +
            self.dataset_info["holdouts"][str(self.settings['holdout'])]["train"]
        ))
        df = df[df['op'].isin(keep_op_range)]

        while True:
            # print(np.random.get_state()[1][0])
            # # Data Augumentation
            if self.call_name == 'train':
                data_augmentation_option = np.random.choice(['rnd', 'dst'])
                if data_augmentation_option == 'dst':
                    this_mask = np.random.randint(0, 51)
                    df_filtered = df.loc[(abs(df['Longitude'] - target_row['Longitude'].values[0]) + abs(df['Latitude'] - target_row['Latitude'].values[0])) >= this_mask, :].copy()
                elif data_augmentation_option == 'rnd':
                    this_mask = np.random.randint(1, len(df) // 2 + 1)
                    indices_to_remove = np.random.choice(df.index, this_mask, replace=False)
                    df_filtered = df.drop(indices_to_remove).copy()
            elif self.call_name == 'test' or self.call_name == 'eval':
                data_augmentation_option = 'dst'
                this_mask = self.mask_distance
                df_filtered = df.loc[(abs(df['Longitude'] - target_row['Longitude'].values[0]) + abs(df['Latitude'] - target_row['Latitude'].values[0])) >= this_mask, :].copy()
            df_filtered = df

            df_filtered.loc[df_filtered['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info[
                "tgt_op"]
            target_row.loc[target_row['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]

            df_known = df_filtered[df_filtered['op'] == self.dataset_info["tgt_op"]].copy()
            df_auxil = df_filtered[df_filtered['op'] != self.dataset_info["tgt_op"]].copy()
            # check and quit loop
            if len(df_known) >= 1 and len(df_auxil) >= 1:
                break

        # turn dataframe to torch tensor according to models
        rtn = getattr(self, f"to_torch_{self.settings['model'].lower()}", None)(df_known, df_auxil, target_row)
        return rtn


def collate_fn(self, examples):
    return getattr(self, f"collate_fn_{self.settings['model'].lower()}", None)(examples)


def to_torch_gnn(self, df_known, df_auxil, target_row):
    graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)

    coords = torch.from_numpy(graph_candidates[['Longitude', 'Latitude']].values).float()
    answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

    full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
    aggregated_df = full_df.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()
    features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
    possible_values = list(self.dataset_info["op_dic"].keys())
    possible_values.sort()

    for op in possible_values:
        aggregated_df_op = aggregated_df[aggregated_df['op'] == op]
        interpolated_grid = torch.zeros((len(graph_candidates), 1))
        if len(aggregated_df_op) != 0:
            xi = graph_candidates['Longitude'].values
            yi = graph_candidates['Latitude'].values
            x = aggregated_df_op['Longitude'].values
            y = aggregated_df_op['Latitude'].values
            values = aggregated_df_op['Result_norm'].values
            interpolated_values = self.idw_interpolation(x, y, values, xi, yi)
            interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
        features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op] + 1] = interpolated_grid

    conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
    features[:, -1] = torch.from_numpy(
        np.where(conditions, graph_candidates['Thing'] / self.dataset_info["non_eu_col"]["Thing"][1],
                 (self.dataset_info['lowest_rank'] - 1) / self.dataset_info["non_eu_col"]["Thing"][1]))
    features = features.float()

    return features, coords, answers


def collate_fn_gnn(self, examples):
    input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
    x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
    c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
    y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)

    return x_b, c_b, y_b, input_lenths


to_torch_pegnn = to_torch_gnn
to_torch_pegat = to_torch_gnn
collate_fn_pegnn = collate_fn_gnn
collate_fn_pegat = collate_fn_gnn


def to_torch_fegnn(self, df_known, df_auxil, target_row):
    graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)

    coords = torch.from_numpy(graph_candidates[['Longitude', 'Latitude']].values).float() / 250.0
    answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

    full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
    aggregated_df = full_df.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()
    features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
    possible_values = list(self.dataset_info["op_dic"].keys())
    possible_values.sort()
    for op in possible_values:
        aggregated_df_op = aggregated_df[aggregated_df['op'] == op]
        interpolated_grid = torch.zeros((len(graph_candidates), 1))
        if len(aggregated_df_op) != 0:
            xi = graph_candidates['Longitude'].values
            yi = graph_candidates['Latitude'].values
            x = aggregated_df_op['Longitude'].values
            y = aggregated_df_op['Latitude'].values
            values = aggregated_df_op['Result_norm'].values
            interpolated_values = self.idw_interpolation(x, y, values, xi, yi)
            interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
        features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op] + 1] = interpolated_grid

    conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
    features[:, -1] = torch.from_numpy(
        np.where(conditions, graph_candidates['Thing'] / self.dataset_info["non_eu_col"]["Thing"][1],
                 (self.dataset_info['lowest_rank'] - 1) / self.dataset_info["non_eu_col"]["Thing"][1]))
    features = features.float()

    trans_fe = torch.concat([features[:, 1:], coords], dim=1)
    gnn_fe = features[:, 0:1]

    return trans_fe, gnn_fe, answers


def collate_fn_fegnn(self, examples):
    input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
    trans_fes = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
    gnn_fes = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
    answers = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)

    return trans_fes, gnn_fes, answers, input_lenths


to_torch_fegnn_fix = to_torch_fegnn
collate_fn_fegnn_fix = collate_fn_fegnn


def norm_fcol(self, df):
    rtn = df.copy()
    # Norm other columns
    for col, (min_val, max_val) in self.dataset_info["eu_col"].items():
        rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
    for col, (min_val, max_val) in self.dataset_info["non_eu_col"].items():
        rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
    return rtn


def to_torch_trsage(self, df_known, df_auxil, target_row):
    df_known = self.norm_fcol(df_known)
    df_auxil = self.norm_fcol(df_auxil)
    target_row = self.norm_fcol(target_row)

    # a_token = target_row[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
    # a_serie = torch.from_numpy(encode_and_bind(a_token, 'op', self.dataset_info["op_dic"]).values).float()

    q_loc_df = target_row[
        ['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + [
            'op', ]].copy()
    q_serie = torch.from_numpy(encode_and_bind(q_loc_df, 'op', self.dataset_info["op_dic"]).values).float()
    # print(q_serie.size())
    q_serie[:, 0] = 0

    answer = torch.from_numpy(target_row[['Result_norm']].values).float()

    df_known = df_known[
        ['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + [
            'op', ]]
    known_serie = torch.from_numpy(encode_and_bind(df_known, 'op', self.dataset_info["op_dic"]).values).float()
    # rc_k = known_serie[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] - q_serie[:, (len(list(self.dataset_info["non_eu_col"].keys()))):(len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))]
    # known_serie_c = known_serie.clone()
    # known_serie_c[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] = rc_k
    # known_serie[:, 1+len(list(self.dataset_info["non_eu_col"].keys())):] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]

    df_auxil = df_auxil[
        ['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + [
            'op', ]]
    auxil_serie = torch.from_numpy(encode_and_bind(df_auxil, 'op', self.dataset_info["op_dic"]).values).float()
    # auxil_serie[:, (1 + len(list(self.dataset_info["non_eu_col"].keys()))) : (1 + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())))] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]
    # rc_a = auxil_serie[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] - q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys()))]
    # auxil_serie_c = auxil_serie.clone()
    # auxil_serie_c[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] = rc_a

    # c = df_known[list(self.dataset_info["eu_col"].keys())].values
    # v = df_known['Result_norm'].values
    # ci = q_loc_df[list(self.dataset_info["eu_col"].keys())].values
    # idw_np = self.idw_interpolation(c, v, ci)
    # idw = torch.from_numpy(idw_np).float()

    # return q_serie, known_serie, auxil_serie, known_serie_c, auxil_serie_c, answer, idw
    return q_serie, known_serie, auxil_serie, answer


def collate_fn_trsage(self, examples):
    q_series = torch.concat([ex[0] for ex in examples], 0)
    known_lenths = torch.tensor([len(ex[1]) for ex in examples])
    auxil_lenths = torch.tensor([len(ex[2]) for ex in examples])
    input_series = pad_sequence([torch.cat([ex[1], ex[2]], dim=0) for ex in examples], batch_first=True,
                                padding_value=0.0)
    # input_series_c = pad_sequence([torch.cat([ex[3], ex[4]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)
    # known_series = pad_sequence([ex[1] for ex in examples], batch_first=True, padding_value=0.0)
    # answers = torch.tensor([ex[5] for ex in examples])
    answers = torch.tensor([ex[3] for ex in examples])
    # idws = torch.tensor([ex[6] for ex in examples])
    # a_series = pad_sequence([torch.cat([ex[7], ex[1], ex[2]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)

    # return q_series, known_lenths, auxil_lenths, input_series, known_series, answers, idws
    return q_series, known_lenths, auxil_lenths, input_series, answers


to_torch_trsage5 = to_torch_trsage
collate_fn_trsage5 = collate_fn_trsage
