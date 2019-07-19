#!/usr/bin/env python

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import List


class AttentionFM(torch.jit.ScriptModule):

    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, attention_dim=-1):
        super(AttentionFM, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.mats = []

        if input_dim > 0 and embedding_dim > 0 and n_fields > 0:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            self.weights = torch.nn.Parameter(torch.zeros(input_dim, 1))
            self.embedding = torch.nn.Parameter(torch.zeros(input_dim, embedding_dim))
            attention_w = torch.nn.Parameter(torch.zeros(embedding_dim, attention_dim))
            attention_b = torch.nn.Parameter(torch.zeros(attention_dim, 1))
            attention_h = torch.nn.Parameter(torch.zeros(attention_dim, 1))
            attention_p = torch.nn.Parameter(torch.zeros(embedding_dim, 1))
            torch.nn.init.xavier_uniform_(self.weights)
            torch.nn.init.xavier_uniform_(self.embedding)
            torch.nn.init.xavier_uniform_(attention_w)
            torch.nn.init.xavier_uniform_(attention_b)
            torch.nn.init.xavier_uniform_(attention_h)
            torch.nn.init.xavier_uniform_(attention_p)
            self.mats = [attention_w, attention_b, attention_h, attention_p]

            self.input_dim = torch.jit.Attribute(self.input_dim, int)
            self.n_fields = torch.jit.Attribute(self.n_fields, int)
            self.embedding_dim = torch.jit.Attribute(self.embedding_dim, int)
            self.mats = torch.jit.Attribute(self.mats, List[Tensor])


    @torch.jit.script_method
    def first_order(self, batch_size, index, values, bias, weights):
        # type: (int, Tensor, Tensor, Tensor, Tensor) -> Tensor
        size = batch_size
        srcs = weights.view(1, -1).mul(values.view(1, -1)).view(-1)
        output = torch.zeros(size, dtype=torch.float32)
        output.scatter_add_(0, index, srcs)
        first = output + bias
        return first

    @torch.jit.script_method
    def second_order(self, batch_size, index, values, embeddings, n_fields, embedding_dim, mats):
        # type: (int, Tensor, Tensor, Tensor, int, int, List[Tensor]) -> Tensor

        attention_w, attention_b, attention_h, attention_p = mats
        biinteraction_num = int(n_fields*(n_fields-1)*0.5)
        embeddings_ = embeddings.view(batch_size, n_fields, embedding_dim)
        tri_indices = torch.triu_indices(n_fields, n_fields, 1)
        indices_i = tri_indices[0]
        indices_j = tri_indices[1]
        biinteraction_result = torch.index_select(embeddings_, 1, indices_i) * torch.index_select(embeddings_, 1, indices_j)

        temp_mul = torch.matmul(biinteraction_result.view(batch_size, biinteraction_num, embedding_dim), attention_w)
        temp_w = torch.relu(temp_mul.view(batch_size, -1) + attention_b.view(-1).repeat(biinteraction_num))

        attention_weight_matrix = F.softmax(torch.matmul(temp_w.view(batch_size, biinteraction_num, -1), attention_h), dim=1)
        attention_weighted_sum = attention_weight_matrix.view(batch_size, biinteraction_num).repeat(1, embedding_dim) * \
                                 biinteraction_result.view(batch_size, -1)
        attention_out = torch.matmul(attention_weighted_sum.view(batch_size, biinteraction_num, -1), attention_p.view(-1)).sum(1)

        return attention_out

    @torch.jit.script_method
    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats):
        # type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor]) -> Tensor

        n_fields = (int)(embeddings.size(0) / batch_size)
        embedding_dim = embeddings.size(1)
        first = self.first_order(batch_size, index, values, bias, weights)
        second = self.second_order(batch_size, index, values, embeddings, n_fields, embedding_dim,mats)

        return torch.sigmoid(first + second)

    @torch.jit.script_method
    def forward(self, batch_size, index, feats, values):
        # type: (int, Tensor, Tensor, Tensor) -> Tensor
        batch_first = F.embedding(feats, self.weights)
        batch_second = F.embedding(feats, self.embedding)
        return self.forward_(batch_size, index, feats, values,
                             self.bias, batch_first, batch_second, self.mats)

    @torch.jit.script_method
    def loss(self, output, targets):
        return self.loss_fn(output, targets)

    @torch.jit.script_method
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.script_method
    def get_name(self):
        return "AttentionFM"


FLAGS = None


def main():
    afm = AttentionFM(FLAGS.input_dim, FLAGS.n_fields, FLAGS.embedding_dim, FLAGS.attention_dim)
    afm.save("attention_fm.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=-1,
        help="data input dim."
    )
    parser.add_argument(
        "--n_fields",
        type=int,
        default=-1,
        help="data num fields."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=-1,
        help="embedding dim."
    )
    parser.add_argument(
        "--attention_dim",
        type=int,
        default=-1,
        help="attention dim."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()

