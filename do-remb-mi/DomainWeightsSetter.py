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
        batch = state.batch

        proxy_logits = state.output.logits
        ref_logits = batch["ref_logits"]
        b, seq_len, _ = proxy_logits.shape

        # TODO: Decide what data domain key is
        # TODO: Implement in a more efficient way
        # Compute the excess loss for the given inputs
        num_tokens = torch.count_nonzero(batch["attention_mask"])
        targets = state.model.get_targets(batch)

        proxy_loss = state.model.loss_fn(
            proxy_logits.view(-1, proxy_logits.logits.size(-1)),
            targets.view(-1))
        ref_loss = state.model.loss_fn(
            ref_logits.view(-1, ref_logits.logits.size(-1)), targets.view(-1))
        excess_loss = torch.maximum(proxy_loss - ref_loss,
                                    torch.zeros_like(proxy_loss))
        excess_loss = excess_loss.view()

        # Compute the updates to the domain weights
        excess_loss = excess_loss.view(b, seq_len)
        excess_loss = torch.sum(excess_loss, dim=-1)
        domain_excess_loss = torch.scatter_reduce(torch.zeros_like(
            self.domain_weights),
                                                  0,
                                                  batch["domain"],
                                                  excess_loss,
                                                  reduce="sum")
        domain_normalization = torch.scatter_reduce(torch.zeros_like(
            self.domain_weights),
                                                    0,
                                                    batch["domain"],
                                                    num_tokens,
                                                    reduce="sum")

        domain_excess_loss = torch.sum(all_gather(domain_excess_loss), dim=0)
        domain_normalization = torch.sum(all_gather(domain_normalization),
                                         dim=0)

        lambdas = domain_excess_loss / domain_normalization
        domain_weights_prime = self.domain_weights * torch.exp(
            self.step_size * lambdas)
        domain_weights = (1 - self.smoothing) * (
            domain_weights_prime / torch.sum(domain_weights_prime)
        ) + self.smoothing * self.domain_weights  # Compute EMA of domain weights

        self.domain_weights = domain_weights
        self.trajectory_domain_weights += domain_weights

        # TODO: Need to add logging of domain weights over trajectory
