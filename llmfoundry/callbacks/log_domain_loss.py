# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from __future__ import annotations

import torch
from composer.core import Callback, State
from composer.loggers import Logger

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]


class LogDomainLoss(Callback):
    """Loss on each data-domain.
    """

    def __init__(self, num_domains: int):
        self.num_domains = num_domains

    # Function assumes `domain_idx` key for each batch
    # Also will only compute for last device_batch
    def batch_end(self, state: State, logger: Logger):
        logits = state.outputs.logits
        b, seq_len, _ = logits.shape

        num_tokens = torch.sum(
            state.batch["input_ids"] != -100, dim=-1
        )  # Might not have attention mask, prob just get ignore toke (-100)
        targets = state.model.get_targets(state.batch)
        print(targets.shape)
        print(targets.view(-1).shape)
        print(logits.shape)
        loss = state.model.proxy_loss_fn(logits.view(-1, logits.size(-1)),
                                         targets.view(-1)).view(b, seq_len)
        loss = torch.sum(loss, dim=-1) / num_tokens

        loss = torch.scatter_reduce(torch.zeros(self.num_domains,
                                                devicex=loss.device,
                                                dtype=loss.dtype),
                                    0,
                                    state.batch["domain_idx"],
                                    loss,
                                    reduce="sum")
        seq_len_normalization = torch.scatter_reduce(torch.zeros(
            self.num_domains, device=num_tokens.device, dtype=num_tokens.dtype),
                                                     0,
                                                     state.batch["domain_idx"],
                                                     num_tokens,
                                                     reduce="sum")

        to_log = {}
        for domain_idx, loss in enumerate(loss / seq_len_normalization):
            loss = loss.cpu().item()
            if loss > 0:
                to_log[
                    f"metrics/domains/domain-{PILE_DATA_SOURCES[domain_idx]}-loss"] = loss
        logger.log_metrics(to_log)
