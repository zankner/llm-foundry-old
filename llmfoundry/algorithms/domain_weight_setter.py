# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the domain weights."""
from __future__ import annotations
import tempfile
import os

import torch
import numpy as np
from composer import Algorithm, Event, State
from composer.utils import dist
from composer.loggers import Logger, RemoteUploaderDownloader

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]


class DomainWeightSetter(Algorithm):

    def __init__(self,
                 num_domains,
                 save_dir,
                 step_size=1.0,
                 smoothing=1e-4,
                 init_dist="uniform",
                 warmup_steps=0,
                 log_domain_weights_freq=100,
                 log_excess_loss=False):

        self.step_size = step_size
        self.smoothing = smoothing

        if init_dist == "uniform":
            self.domain_weights = torch.ones(num_domains) / num_domains
        else:
            # self.domain_weights = init_dist
            raise NotImplementedError(
                "Only uniform initialization is supported")

        self.trajectory_domain_weights = self.domain_weights
        self.num_updates = 1
        self.smoothing_dist = torch.ones(num_domains) / num_domains
        self.lambdas = torch.zeros(num_domains)

        self.warmup_steps = warmup_steps

        self.save_dir = save_dir
        self.log_domain_weights_freq = log_domain_weights_freq
        self.log_excess_loss = log_excess_loss

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.BEFORE_FORWARD or event == Event.AFTER_FORWARD or
                event == Event.FIT_END or event == Event.AFTER_TRAIN_BATCH)

    def _upload_data(self, data, path: str, uploader: RemoteUploaderDownloader,
                     state: State):
        with tempfile.NamedTemporaryFile() as tmp_file:
            np.save(tmp_file.name + ".npy", data)
            uploader.upload_file(
                state=state,
                remote_file_name=os.path.join(self.save_dir, path),
                file_path=tmp_file.name + ".npy",
                overwrite=True
            )  # Overwrite to handle mismatch between weight freq and ckpt freq

    def _log_domain_weights(self, state: State, final: bool = False) -> None:
        if dist.get_global_rank() == 0:
            remote_uploader = [
                callback for callback in state.callbacks
                if (isinstance(callback, RemoteUploaderDownloader) and callback.
                    backend_kwargs['bucket'] == "mosaicml-internal-doremi")
            ]  # Ugly but oh well for now, change later
            assert len(
                remote_uploader) == 1, "Only one remote uploader is supported"
            remote_uploader = remote_uploader[0]

            if (int(state.timestamp.batch_in_epoch) +
                    1) % self.log_domain_weights_freq != 0 and int(
                        state.timestamp.batch) != 0 and not final:
                return

            if final:
                prefix = "final"
            elif int(state.timestamp.batch) == 0:
                prefix = f"ba-{int(state.timestamp.batch)}"
            else:
                prefix = f"ba-{int(state.timestamp.batch) + 1}"
            domain_weights_path = os.path.join(prefix, "domain_weights.npy")
            average_domain_weights_path = os.path.join(
                prefix, "average_domain_weights.npy")
            self._upload_data(self.domain_weights.cpu().numpy(),
                              path=domain_weights_path,
                              uploader=remote_uploader,
                              state=state)
            num_steps = max(1, self.num_updates + 1 - self.warmup_steps)
            average_domain_weights = self.trajectory_domain_weights / num_steps
            self._upload_data(average_domain_weights.cpu().numpy(),
                              path=average_domain_weights_path,
                              uploader=remote_uploader,
                              state=state)

    @torch.no_grad()
    def apply(self, event: Event, state: State, logger: Logger) -> None:
        device = state.batch["input_ids"].device
        self.smoothing_dist = self.smoothing_dist.to(device)
        self.domain_weights = self.domain_weights.to(device)
        self.lambdas = self.lambdas.to(device)
        self.trajectory_domain_weights = self.trajectory_domain_weights.to(
            device)

        if event == Event.BEFORE_FORWARD:
            if int(state.timestamp.batch) != 0:
                return
            device = state.batch["input_ids"].device
            self.domain_weights = self.domain_weights.to(device)
            self.trajectory_domain_weights = self.trajectory_domain_weights.to(
                device)
            state.batch_set_item(key="domain_weights",
                                 value=self.domain_weights)

            self._log_domain_weights(state)
        elif event == Event.FIT_END:
            self._log_domain_weights(state, final=True)
        elif event == Event.AFTER_FORWARD:
            domain_excess_loss, ref_loss, proxy_loss, seq_len_normalization = state.model.compute_domain_wise_excess_loss(
                state.outputs,
                state.batch,
                self.domain_weights,
                non_zero_excess=True)

            dist.all_reduce(ref_loss, "sum")
            dist.all_reduce(proxy_loss, "sum")
            dist.all_reduce(domain_excess_loss, "sum")
            dist.all_reduce(seq_len_normalization, "sum")
            dist.barrier()

            seq_len_normalization = torch.maximum(
                seq_len_normalization, torch.ones_like(seq_len_normalization)
            )  # Avoid changing unused domain weights

            if self.log_excess_loss and dist.get_global_rank() == 0:
                to_log = {}
                for domain_idx, (excess_loss, ref_loss,
                                 proxy_loss) in enumerate(
                                     zip(
                                         domain_excess_loss /
                                         seq_len_normalization,
                                         ref_loss / seq_len_normalization,
                                         proxy_loss / seq_len_normalization)):
                    excess_loss = excess_loss.cpu().item()
                    ref_loss = ref_loss.cpu().item()
                    proxy_loss = proxy_loss.cpu().item()
                    if excess_loss > 0:
                        to_log[
                            f"Excess-loss/domain-{PILE_DATA_SOURCES[domain_idx]}"] = excess_loss
                        to_log[
                            f"Ref-loss/domain-{PILE_DATA_SOURCES[domain_idx]}"] = ref_loss
                        to_log[
                            f"Proxy-loss/domain-{PILE_DATA_SOURCES[domain_idx]}"] = proxy_loss
                logger.log_metrics(to_log)

            if int(state.timestamp.batch) < self.warmup_steps:
                state.batch_set_item(key="domain_weights",
                                     value=self.domain_weights)
                self._log_domain_weights(state)
                return

            lambdas = domain_excess_loss / seq_len_normalization
            lambdas[(lambdas == 0.0)] = self.lambdas[(lambdas == 0.0)]
            self.lambdas = lambdas

            domain_weights_prime = self.domain_weights * torch.exp(
                self.step_size * lambdas)
            domain_weights_prime = domain_weights_prime / torch.sum(
                domain_weights_prime)  # Normalizing domain weight update

            domain_weights = (
                1 - self.smoothing
            ) * domain_weights_prime + self.smoothing * self.smoothing_dist  # Compute EMA of domain weights

            self.domain_weights = domain_weights
            self.trajectory_domain_weights += domain_weights
            self.num_updates += 1

            state.batch_set_item(key="domain_weights",
                                 value=self.domain_weights)

            self._log_domain_weights(state)

        dist.barrier()
