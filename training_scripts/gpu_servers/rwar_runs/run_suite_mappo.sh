#!/bin/bash

# This script queues jobs for all seeds, acting policies, and envs
# bash ./training_scripts/gpu_servers/rwar_runs/run_suite_mappo.sh
seeds=(6342 1754 3388 4985 7301)
gpus=(2 3 4 5 6 7)
acting_policies=("single" "hyperbolic" "average" "largest")
run_name="rwar_mappo_mh"
wandb_project="mh-marl"
wandb_group="mappo"
# List of envs to run for number of seeds
envs=("rware:rware-tiny-4ag-v1" "rware:rware-tiny-2ag-v1" "rware:rware-small-4ag-v1" "rware:rware-small-2ag-v1")

for env_key in "${envs[@]}"; do
  for acting_policy in "${acting_policies[@]}"; do
    for seed in "${seeds[@]}"; do
      while true; do
        for gpu_id in "${gpus[@]}"; do
          nvidia_smi_output=$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader)
          process_count=$(echo "$nvidia_smi_output" | awk '$1!="" {print $1}' | wc -l)
          if [[ $process_count -lt 3 ]]; then
            echo $process_count
            gpu="--gpus device=$gpu_id"
            env_key_short="$(echo "$env_key" | awk -F: '{print $2}')"
            name="${run_name}_${env_key_short}_${acting_policy}_${seed}_gpu-${gpu_id}"
            use_mh=True
            if [[ "$acting_policy" == "single" ]]; then
              use_mh=False
            fi
            wandb docker-run --name "$name" $gpu -ti -d --rm kevduong1/mh-marl:mh \
              --config=mappo --env-config=gymma with \
              hidden_dim=128 lr=0.0005 entropy_coef=0.001 q_nstep=10 \
              target_update_interval_or_tau=0.01 use_rnn=False use_mh=$use_mh num_gammas=3 hyp_exp=0.1 acting_policy="$acting_policy" \
              standardise_rewards=False standardise_returns=False \
              env_args.time_limit=500 t_max=20050000 env_args.key="$env_key" \
              use_wandb=True wandb_args.project="$wandb_project" wandb_args.group="$wandb_group" wandb_args.tag="$acting_policy" seed="$seed"
            sleep 10
            break 2
          fi
        done
        sleep 5
      done
    done
  done
done
