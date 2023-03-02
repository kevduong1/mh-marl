import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.mh_mixers.qmix import QMixer

# Fully multi-horizon QMixer
# We create a full mixing network for each gamma
class Full_MH_QMixer(nn.Module):
    def __init__(self, args):
        super(Full_MH_QMixer, self).__init__()
        
        # Number of gammas for multi-horizon
        self.num_gammas = args.num_gammas

        # Create a seperate QMixer for each gamma
        self.MH_QMIXERS = nn.ModuleList([QMixer(args) for _ in range(self.num_gammas)])

    def forward(self, agent_qs, states):
        q_tot = [None] * self.num_gammas
        for i in range(self.num_gammas):
            q_tot[i] = self.MH_QMIXERS[i](agent_qs, states)
        
        return q_tot
