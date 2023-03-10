import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mh_mixers.qmix import QMixer
import torch as th
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qmix":
                # Creating mixers for each gamma q's (if using mh)
                if args.use_mh:
                    self.mixer = [None] * args.num_gammas
                    self.target_mixer = [None] * args.num_gammas
                    for i in range(args.num_gammas):
                        self.mixer[i] = QMixer(args)
                        self.params += list(self.mixer[i].parameters())
                        self.target_mixer[i] = copy.deepcopy(self.mixer[i])
                else:
                    self.mixer = [QMixer(args)]
                    self.params += list(self.mixer[0].parameters())
                    self.target_mixer = [copy.deepcopy(self.mixer[0])]
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    # TODO: (Important) Need to test this after merging the two functions for simplicity
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1]
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            if self.args.standardise_rewards:
                self.rew_ms.update(rewards)
                rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
            
            # forward function for mh provides a list of agent outs for each gamma value
            agent_outs_list = {}
            target_agent_outs_list = {}
            self.mac.init_hidden(batch.batch_size)
            self.target_mac.init_hidden(batch.batch_size)


            # Get agent outs and store in relative t dictionary (check for t in range loops below)
            if self.args.use_mh:
                gammas = self.mac.gammas
                for t in range(batch.max_seq_length):
                    agent_outs_list[t] = self.mac.forward(batch, t=t, learning_mode=True)
                    target_agent_outs_list[t] = self.target_mac.forward(batch, t=t, learning_mode=True)
            else:
                # use single gamma for single-horizon learning (place gamma and outs in list for indexing logic that works with mh)
                gammas = [self.args.gamma] 
                for t in range(batch.max_seq_length):
                    agent_outs_list[t] = [self.mac.forward(batch, t=t, learning_mode=True)]
                    target_agent_outs_list[t] = [self.target_mac.forward(batch, t=t, learning_mode=True)]

            total_loss = None
            # Calculate loss for each gamma head output
            for i, gamma in zip(range(len(gammas)), gammas):
                # Calculate estimated Q-Values
                mac_out = []
                for t in range(batch.max_seq_length):
                    agent_outs = agent_outs_list[t][i]
                    mac_out.append(agent_outs)
                mac_out = th.stack(mac_out, dim=1)  # Concat over time
                # Pick the Q-Values for the actions taken by each agent
                chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

                # Calculate the Q-Values necessary for the target
                target_mac_out = []
                for t in range(batch.max_seq_length):
                    target_agent_outs = target_agent_outs_list[t][i]
                    target_mac_out.append(target_agent_outs)

                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

                # Mask out unavailable actions
                target_mac_out[avail_actions[:, 1:] == 0] = -9999999

                # Max over target Q-Values
                if self.args.double_q:
                    # Get actions that maximise live Q (for double q-learning)
                    mac_out_detach = mac_out.clone().detach()
                    mac_out_detach[avail_actions == 0] = -9999999
                    cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                    target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                else:
                    target_max_qvals = target_mac_out.max(dim=3)[0]

                # Mix
                if self.mixer is not None:
                    chosen_action_qvals = self.mixer[i](chosen_action_qvals, batch["state"][:, :-1])
                    target_max_qvals = self.target_mixer[i](target_max_qvals, batch["state"][:, 1:])

                if self.args.standardise_returns:
                    target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

                # Calculate 1-step Q-Learning targets
                targets = rewards + gamma * (1 - terminated) * target_max_qvals.detach()

                if self.args.standardise_returns:
                    self.ret_ms.update(targets)
                    targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

                # Td-error
                td_error = (chosen_action_qvals - targets.detach())

                mask = mask.expand_as(td_error)

                # 0-out the targets that came from padded data
                masked_td_error = td_error * mask

                # Normal L2 loss, take mean over actual data and total up losses for each gamma head
                if total_loss == None:
                    total_loss = (masked_td_error ** 2).sum() / mask.sum()
                else:
                    total_loss += (masked_td_error ** 2).sum() / mask.sum()

            # preserve scale of loss according to num gammas
            total_loss /= len(gammas)
            # Optimise
            self.optimiser.zero_grad()
            total_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

            self.training_steps += 1
            if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_step = self.training_steps
            elif self.args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.args.target_update_interval_or_tau)

            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat("loss", total_loss.item(), t_env)
                self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
                self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            for i in range(len(self.mixer)):
                self.target_mixer[i].load_state_dict(self.mixer[i].state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for i in range(len(self.mixer)):
                for target_param, param in zip(self.target_mixer[i].parameters(), self.mixer[i].parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            for i in range(len(self.mixer)):
                self.mixer[i].cuda()
                self.target_mixer[i].cuda()
            # self.mixer.cuda()
            # self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            # TODO: Will need to save these in list form
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            # TODO: Will need to load these in list form
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
