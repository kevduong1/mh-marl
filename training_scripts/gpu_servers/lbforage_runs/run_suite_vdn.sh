#!/bin/bash

# This script queues jobs for all seeds, acting policies, and envs, and then 
# bash ./training_scripts/gpu_servers/lbforage_runs/run_suite_vdn.sh
seeds=(6342 1754 3388 4985 7301)
gpus=(0 1)
acting_policies=("single" "hyperbolic" "largest")
run_name="lbforaging_vdn"
wandb_project="mh-marl"
wandb_group="vdn"
# List of envs to run for number of seeds
envs=("lbforaging:Foraging-15x15-4p-3f-v2" "lbforaging:Foraging-15x15-3p-5f-v2" "lbforaging:Foraging-15x15-4p-5f-v2")

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
              --config=vdn --env-config=gymma with \
              hidden_dim=128 lr=0.0003 evaluation_epsilon=0.0 epsilon_anneal_time=200000 \
              target_update_interval_or_tau=0.01 use_rnn=True use_mh=$use_mh num_gammas=3 hyp_exp=0.1 acting_policy="$acting_policy" \
              standardise_rewards=True standardise_returns=False \
              env_args.time_limit=25 t_max=2050000 env_args.key="$env_key" \
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
