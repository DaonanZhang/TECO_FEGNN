import wandb
import subprocess
import os
import json
import time
import myconfig_al_only as myconfig
from datetime import datetime


# check echo from 'sacct' to tell the job status
def check_status(status):
    rtn = 'RUNNING'
    
    lines = status.split('\n')
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        if 'FAILED' in line:
            rtn = 'FAILED'
            break
        elif 'COMPLETED' not in line:
            rtn = 'PENDING'
            break
    else:
        rtn = 'COMPLETED'
        
    return rtn


def wrap_task(config=None):
    # recieve config for this run from Sweep Controller

    with wandb.init(config=config):
        agent_id = wandb.run.id
        agent_dir = wandb.run.dir
        config = dict(wandb.config)

        # §§ DB folder
        config['origin_path'] = './Dataset_res250_reg4c/'

        config['bp'] = False

        # optimized hyperparameters
        config['full_batch'] = 128
        config['conv_dim'] = 256
        config['emb_dim'] = 32
        config['lr'] = 1e-5

        
        config['batch'] = 16
        config['accumulation_steps'] = config['full_batch'] // config['batch']

        config['test_batch'] = 1
        config['k'] = 20

        config['nn_lr'] = 1e-5
        config['lr'] = 1e-5
        
        config['es_mindelta'] = 0.5
        config['es_endure'] = 5

        # ！！！
        config['num_features_in'] = 9
        
        config['num_features_out'] = 1

        config['emb_hidden_dim'] = 256
        
        config['model'] = 'PEGNN'
        config['fold'] = 4
        config['lowest_rank'] = 1

        config['hp_marker'] = 'tuned'
        config['nn_length'] = 3
        config['nn_hidden_dim'] = 32
        config['dropout_rate'] = 0.1

        # for transformer
        config['nhead'] = 2
        config['d_model'] = 32
        config['num_encoder_layers'] = 2
        config['env_features_in'] = 11
        config['transformer_dropout'] = 0.1
        config['dim_feedforward'] = 128
        

        config['epoch'] = 20
        config['debug'] = False

        # # # debug mode
        # config['epoch'] = 4
        # config['debug'] = True
        
        # --------AL mode--------
        config['aux_task_num'] = 1
        config['hyper_lr'] = 1e-5
        config['hyper_decay'] = 0.0
        config['hyper_pre'] = -1
        config['hyper_interval'] = 100
        config['hyper_aux_loss_weight'] = 0.1

        config['emb_dim'] = 16
        config['transformer_dec_output'] = 32

         # ----Task heads----
        config['heads_nn_length'] = 2
        config['heads_nn_hidden_dim'] = 64
        config['heads_dropout_rate'] = 0.1


        # §§ slurm command: squeue
        while True:
            cmd = f"squeue -n {myconfig.project_name}"
            status = subprocess.check_output(cmd, shell=True).decode()
            lines = status.split('\n')[1:-1]
            if len(lines) <= myconfig.pool_size:
                break
            else:
                time.sleep(60)

        # partition gpu_4 => dev_gpu_4
        # time = 24:00:00 => 00:30:00
        # then build up the slurm script

        job_script = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --partition=single
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120000
#SBATCH --error={myconfig.log_path}%x.%j.err
#SBATCH --output={myconfig.log_path}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL
#SBATCH --time=20:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID
python {myconfig.train_script_name} $job_id '{json.dumps(config)}' {agent_id} {agent_dir}
"""
        # wandb config agent id agent dir
        
        # Write job submission script to a file
        # change to windows cmd
        with open(myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch", "w") as f:
            f.write(job_script)

        # current_direcorty =os.getcwd()
        # slurm_scripts_diretocry = os.path.join(current_direcorty, myconfig.slurm_scripts_path)
        # with open( slurm_scripts_diretocry + f"{wandb.run.id}.cmd", "w") as f:
        #     f.write(job_script)
        
        # Submit job to Slurm system and get job ID

        # change script to windows cmd
        cmd = "sbatch " + myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch"
        # cmd = slurm_scripts_diretocry + f"{wandb.run.id}.cmd"

        # change to windows cmd
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        # subprocess.run([cmd] , shell=True)

        # close for now
        job_id = output.split()[-1]

        wandb.log({
            "job_id" : job_id,
        })
        return job_id
        
           
if __name__ == '__main__':
    rtn = wrap_task()
    print(f'******************************************************* Process Finished with code {rtn}')
    wandb.finish()