# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the domain weights."""
from __future__ import annotations
from composer import State

import torch
from composer.core import Callback, State
from composer.utils.dist import all_gather
from composer.loggers import Logger


class DomeainWeightSetter(Callback):

    def __init__(self,
                 num_domains,
                 step_size=1.0,
                 smoothing=1e-4,
                 init_dist=None):

        self.step_size = step_size
        self.smoothing = smoothing

        if init_dist:
            self.domain_weights = init_dist
        else:
            self.domain_weights = torch.ones(num_domains) / num_domains
        self.trajectory_domain_weights = self.domain_weights

    @torch.no_grad()
    def after_forward(self, state: State, logger: Logger) -> None:
        domain_excess_loss, seq_len_normalization = self.model.compute_domain_wise_excess_loss(
            state.outputs,
            state.batch,
            self.domain_weights,
            non_zero_excess=True)

        domain_excess_loss = torch.sum(all_gather(domain_excess_loss), dim=0)
        seq_len_normalization = torch.sum(all_gather(seq_len_normalization),
                                          dim=0)
        seq_len_normalization = torch.maximum(
            seq_len_normalization, torch.ones_like(
                seq_len_normalization))  # Avoid changing unused domain weights

        lambdas = domain_excess_loss / seq_len_normalization
        domain_weights_prime = self.domain_weights * torch.exp(
            self.step_size * lambdas)
        domain_weights_prime = domain_weights_prime / torch.sum(
            domain_weights_prime)  # Normalizing domain weight update

        domain_weights = (1 - self.smoothing) * (
            domain_weights_prime / torch.sum(domain_weights_prime)
        ) + self.smoothing * self.domain_weights  # Compute EMA of domain weights

        self.domain_weights = domain_weights
        self.trajectory_domain_weights += domain_weights

        state.batch_set_item(key="domain_weights", value=self.domain_weights)

        # TODO: Need to add logging of domain weights over trajectory