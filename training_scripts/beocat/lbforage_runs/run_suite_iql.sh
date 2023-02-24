#!/bin/bash

#SBATCH --job-name=lbforage_runs
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=05:00:00

# This script queues jobs for all seeds, acting policies, and envs
# bash ./training_scripts/beocat/lbforage_runs/run_suite_iql2.sh

wandb_project="mh-marl"
wandb_group="iql_new_seed"

# List of envs to run for number of seeds
# IMPORTANT CALCULATE THE NUMBER OF COMBINATIONS AND ADJUST JOB ARRAY ID by 2 round up to nearest even
seeds=(49464 78691 49461 14757 93942)
acting_policies=("single" "average" "hyperbolic" "largest")
envs=("lbforaging:Foraging-15x15-4p-3f-v2" "lbforaging:Foraging-15x15-3p-5f-v2" "lbforaging:Foraging-15x15-4p-5f-v2")


# Number of acting policies, seeds, and envs
num_policies=${#acting_policies[@]}
num_seeds=${#seeds[@]}
num_envs=${#envs[@]}
combinations=$((num_envs * num_policies * num_seeds))

# Calculate the current index from the SLURM_ARRAY_TASK_ID
index=$((SLURM_ARRAY_TASK_ID * 2))
env_index=$((index % num_envs))
acting_policy_index=$((index / num_envs % num_policies))
seed_index=$((index / (num_envs * num_policies)))

env_key_1=${envs[$env_index]}
acting_policy_1=${acting_policies[$acting_policy_index]}
seed_1=${seeds[$seed_index]}

use_mh=True
if [[ "$acting_policy_1" == "single" ]]; then
  use_mh=False
fi

index=$((index + 1))
if [ "$index" -lt "$combinations" ]; then
env_index=$((index % num_envs))
acting_policy_index=$((index / num_envs % num_policies))
seed_index=$((index / (num_envs * num_policies)))

env_key_2=${envs[$env_index]}
acting_policy_2=${acting_policies[$acting_policy_index]}
seed_2=${seeds[$seed_index]}

use_mh_2=True
if [[ "$acting_policy_2" == "single" ]]; then
  use_mh_2=False
fi
python3 src/main.py --config=iql --env-config=gymma with \
              hidden_dim=128 lr=0.0003 evaluation_epsilon=0.05 epsilon_anneal_time=200000 \
              target_update_interval_or_tau=200 use_rnn=True use_mh=$use_mh num_gammas=3 hyp_exp=0.1 acting_policy="$acting_policy_1" \
              standardise_rewards=True standardise_returns=False \
              env_args.time_limit=25 t_max=2050000 env_args.key="$env_key_1" \
              use_wandb=True wandb_args.project="$wandb_project" wandb_args.group="$wandb_group" wandb_args.tag="$acting_policy_1" seed="$seed_1" &

python3 src/main.py --config=iql --env-config=gymma with \
              hidden_dim=128 lr=0.0003 evaluation_epsilon=0.05 epsilon_anneal_time=200000 \
              target_update_interval_or_tau=200 use_rnn=True use_mh=$use_mh_2 num_gammas=3 hyp_exp=0.1 acting_policy="$acting_policy_2" \
              standardise_rewards=True standardise_returns=False \
              env_args.time_limit=25 t_max=2050000 env_args.key="$env_key_2" \
              use_wandb=True wandb_args.project="$wandb_project" wandb_args.group="$wandb_group" wandb_args.tag="$acting_policy_2" seed="$seed_2" 
else
python3 src/main.py --config=iql --env-config=gymma with \
              hidden_dim=128 lr=0.0003 evaluation_epsilon=0.05 epsilon_anneal_time=200000 \
              target_update_interval_or_tau=200 use_rnn=True use_mh=$use_mh num_gammas=3 hyp_exp=0.1 acting_policy="$acting_policy_1" \
              standardise_rewards=True standardise_returns=False \
              env_args.time_limit=25 t_max=2050000 env_args.key="$env_key_1" \
              use_wandb=True wandb_args.project="$wandb_project" wandb_args.group="$wandb_group" wandb_args.tag="$acting_policy_1" seed="$seed_1" &
fi

wait