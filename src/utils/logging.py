from collections import defaultdict
import logging
import numpy as np

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.wandb = None

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = False # Disabling sacred since we use wandb
    
    def setup_wandb(self, configs):
        import wandb
        if configs["use_mh"] == False:
            configs["discounting_policy"] = "single"
            configs["num_gammas"] = 1
            configs["hyp_exp"] = None
            configs["integral_estimate"] = None

        run_name = "{}_{}_seed:{}".format(configs["name"],configs["discounting_policy"], configs["env_args"]["seed"])

        self.wandb = wandb.init(group=configs["wandb_args"]["group"], name=run_name, project=configs["wandb_args"]["project"], entity="kdd-drl", config=configs,
                                tags=[configs["wandb_args"]["tag"]] if configs["wandb_args"]["tag"] else None )
        
    def finish(self):
        if self.wandb != None:
            self.wandb.finish()

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]
            
            self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
        
    def log_wandb_stats(self):
        log = {}
        for (k, v) in sorted(self.stats.items()):
            window = 5 if k != "epsilon" else 1
            try:
                item = np.mean([x[1] for x in self.stats[k][-window:]]).item()
            except:
                item = np.mean([x[1].item() for x in self.stats[k][-window:]]).item()
            
            log[k] = item

        total_steps = self.stats["episode"][-1][0]
        self.wandb.log(log, step=total_steps)
        


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

