import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        # If we're using multi-horizon learning, create gamma head for each output
        if args.use_mh:
            for i in range(self.args.num_gammas):
                setattr(self, "gamma_head_{}".format(i),nn.Linear(args.hidden_dim, args.n_actions),)
        else:
            self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        
        # Compute q_val for each gamma head, then output as list
        if self.args.use_mh:
            q_vals = [None] * self.args.num_gammas
            for i in range(self.args.num_gammas):
                q_vals[i] = getattr(self, 'gamma_head_{}'.format(i))(h)
            return q_vals, h
        else:
            q = self.fc2(h)
            return q, h
