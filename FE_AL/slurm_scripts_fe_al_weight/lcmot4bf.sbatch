#!/bin/bash
#SBATCH --job-name=fe_al_weight
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:2
#SBATCH --error=./logs_fe_al_weight/%x.%j.err
#SBATCH --output=./logs_fe_al_weight/%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=detecter01@proton.me
#SBATCH --export=ALL
#SBATCH --time=30:00:00

eval "$(conda shell.bash hook)"
conda activate /home/kit/stud/uqqww/miniconda3/envs/ml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kit/stud/uqqww/miniconda3/envs/ml/lib

job_id=$SLURM_JOB_ID
python ./Trainer_fe_al.py $job_id '{"holdout": 3, "hyper_aux_loss_weight": 0.3, "origin_path": "./Dataset_res250_reg4c/", "bp": false, "full_batch": 128, "conv_dim": 256, "emb_dim": 16, "lr": 1e-05, "batch": 8, "accumulation_steps": 16, "test_batch": 1, "k": 20, "nn_lr": 1e-05, "es_mindelta": 0.5, "es_endure": 5, "num_features_in": 2, "num_features_out": 1, "emb_hidden_dim": 256, "model": "PEGNN", "fold": 4, "lowest_rank": 1, "hp_marker": "tuned", "nn_length": 3, "nn_hidden_dim": 32, "dropout_rate": 0.1, "nhead": 2, "d_model": 32, "num_encoder_layers": 2, "env_features_in": 11, "transformer_dropout": 0.1, "dim_feedforward": 128, "epoch": 20, "debug": false, "seed": 1, "aux_task_num": 1, "hyper_lr": 1e-05, "hyper_decay": 0.0, "hyper_pre": -1, "hyper_interval": 100, "transformer_dec_output": 32, "heads_nn_length": 2, "heads_nn_hidden_dim": 64, "heads_dropout_rate": 0.1}' lcmot4bf /pfs/data5/home/kit/stud/uqqww/PEGNN_reg4/wandb/run-20240716_103646-lcmot4bf/files
