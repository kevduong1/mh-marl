#!/bin/bash

python src/main.py \
    --config=mhmix --env-config=gymma with \
    hidden_dim=64 lr=0.0003 evaluation_epsilon=0.05 epsilon_anneal_time=200000 \
    target_update_interval_or_tau=10 use_rnn=True use_mh=False num_gammas=3 hyp_exp=0.1 discounting_policy="single" \
    standardise_rewards=True standardise_returns=False \
    test_interval=10000 log_interval=10000 \
    runner_log_interval=2000 learner_log_interval=2000 \
    env_args.time_limit=25 t_max=2050000 env_args.key="lbforaging:Foraging-15x15-4p-3f-v2" \
    use_wandb=False wandb_args.project="f" wandb_args.group="f" wandb_args.tag="f" seed="12"