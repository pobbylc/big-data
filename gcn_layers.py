import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, nrelations, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_relation = nrelations

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.w_relation = Parameter(torch.FloatTensor(nrelations))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.w_relation.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        # print(input.shape, self.weight.shape)
        '''
        adj(nstock, nstock) * input(nstock, nfeat) * weight(nfeat, nhid)

        adj * w_relation * input * weight
        (nstock, nstock, nrelation) (nrelation) (nstock, nfeat) (nfeat, nhid)
        '''
        # print(adj.shape, self.w_relation.shape)
        support = torch.matmul(input, self.weight)
        if self.n_relation != 1:
            adj = torch.matmul(adj, self.w_relation)
        output = torch.matmul(adj, support)
        # print(support.shape)
        # print(output.shape)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
