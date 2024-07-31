# wandb api key
# §§
# api_key = '50d21c78c93763fe3ab0c50a896d798f9a8e7d0a'

api_key = '8407bdbf6e108552607bce672dbb7c285ce4b172'

# is it a new run or continue
new_run = True

# sweep id still required
# currently not used, used for sweep in wandb
# §§
sweep_id = ''


# where should the intermedia generated scripts be saved (automatically cleaned at the start of each run)
slurm_scripts_path = './slurm_scripts/'

# where should the outputs & logs be saved (automatically cleaned at the start of each run)
log_path = './logs/'
# where should calculation nodes save their important results (e.g. best model weights)
coffer_path = './coffer_idw_final/'

entity_name = 'daonan_'

# project name on wandb and HPC
project_name = 'PEGNN_reg4_PE_sweep'
# e-mail address to recieve notifications

e_mail = 'detecter01@proton.me'

conda_env = '/home/kit/stud/uqqww/miniconda3/envs/ml'

slurm_wrapper_name = './slurm_wrapper_cpu.py'

train_script_name = './Trainer_pe_only.py'

# define custom sweep hyperparameters
#     - how many sweeps do you want to run in total
# 243 combinations for total_sweep
total_sweep = 20
#     - how many sweeps do you want to run parallelly
pool_size = 20


# define wandb sweep parameters
#     - project definition
sweep_config = {
    "project": project_name,
    'program': slurm_wrapper_name,
    "name": "offline-sweep",
    'method': 'random',
}
#     - metric definition
metric = {
    'name': 'best_err',
    'goal': 'minimize'   
}
sweep_config['metric'] = metric
#     - parameters search range definition
parameters_dict = {
    'fold': {
        'values': [0,1,2,3],
    },
    'seed': {
        'values': [1,2,3,4,5]  
    },
    
}
sweep_config['parameters'] = parameters_dict
