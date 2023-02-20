wandb docker-run --gpus device=2 -ti kevduong1/mh-marl:1.0 \
    --config=iql_ns --env-config=pettingzoo with \
    epsilon_anneal_time=5000000 \
    hidden_dim=128 lr=0.0005 target_update_interval_or_tau=0.01 \
    env_args.time_limit=500 t_max=10000000 env_args.key="None" \
    wandb_args.project="mh-v1" wandb_args.tag="idql_ns" wandb_args.group="kaz" seed=10002