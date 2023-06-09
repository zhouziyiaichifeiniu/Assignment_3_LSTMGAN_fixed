from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.gx = nn.Linear(input_dim, hidden_dim, bias=True).cuda()
        self.gh = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
        self.ix = nn.Linear(input_dim, hidden_dim, bias=True).cuda()
        self.ih = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
        self.fx = nn.Linear(input_dim, hidden_dim, bias=True).cuda()
        self.fh = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
        self.ox = nn.Linear(input_dim, hidden_dim, bias=True).cuda()
        self.oh = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
        self.bg = torch.zeros(batch_size, hidden_dim).cuda()
        self.bi = torch.zeros(batch_size, hidden_dim).cuda()
        self.bf = torch.zeros(batch_size, hidden_dim).cuda()
        self.bo = torch.zeros(batch_size, hidden_dim).cuda()
        self.bp = torch.zeros(batch_size, output_dim).cuda()
        self.h = torch.zeros(self.hidden_dim, self.hidden_dim).cuda()
        self.c = torch.zeros(self.hidden_dim, self.hidden_dim).cuda()
        self.ph = nn.Linear(hidden_dim, output_dim, bias=True).cuda()
        self.tanh = nn.Tanh().cuda()
        self.sigmoid = nn.Sigmoid().cuda()


    def forward(self, x):
        # Implementation here ...
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_length, self.input_dim)
        for i in range(self.seq_length):
            # 每个时间单独计算返回输出
            for t in range(self.seq_length):
                g = self.tanh(self.gx(x[:, t]) + self.gh(self.h) + self.bg)
                i = self.sigmoid(self.ix(x[:, t]) + self.ih(self.h) + self.bi)
                f = self.sigmoid(self.fx(x[:, t]) + self.fh(self.h) + self.bf)
                o = self.sigmoid(self.ox(x[:, t]) + self.oh(self.h) + self.bo)
                self.c = g * i + self.c * f
                self.h = torch.tanh(self.c) * o
            return self.ph(self.h) + self.bp


    # add more methods here if needed