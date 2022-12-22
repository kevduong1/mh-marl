#!/bin/bash

wandb docker-run --gpus device=0 -ti mh-marl:1.0 \
    --config=mappo_ns --env-config=gymma with \
    hidden_dim=128 lr=0.0005 entropy_coef=0.001 target_update_interval_or_tau=0.01 \
    env_args.time_limit=500 t_max=5000000 env_args.key="rware:rware-tiny-2ag-v1" \
    wandb_args.project="mh-v1" wandb_args.tag="baseline" wandb_args.group="single" seed=10002