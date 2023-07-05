# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the domain weights."""
from __future__ import annotations
import tempfile
import os
import re
from typing import Any, Dict

import torch
import numpy as np
from composer import Algorithm, Event, State
from composer.utils import dist, get_file
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
                 warmup_steps=0,
                 doremi_iter=1,
                 log_domain_weights_freq=100,
                 log_excess_loss=False):

        self.save_dir = save_dir
        self.step_size = step_size
        self.smoothing = smoothing

        if doremi_iter == 1:
            self.domain_weights = torch.ones(num_domains) / num_domains
        elif doremi_iter > 1:
            self.domain_weights = self._load_domain_weights(doremi_iter)
            print(self.domain_weights)
        else:
            raise ValueError("DoReMi iteration must be >= 1")

        self.trajectory_domain_weights = self.domain_weights
        self.num_updates = 1
        self.smoothing_dist = torch.ones(num_domains) / num_domains
        self.lambdas = torch.zeros(num_domains)

        # Handling grad accumulation
        self.domain_excess_loss = torch.zeros(num_domains)
        self.seq_len_normalization = torch.zeros(num_domains)

        self.warmup_steps = warmup_steps

        self.log_domain_weights_freq = log_domain_weights_freq
        self.log_excess_loss = log_excess_loss

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["domain_weights"] = self.domain_weights
        state_dict["trajectory_domain_weights"] = self.trajectory_domain_weights
        state_dict["num_updates"] = self.num_updates
        state_dict["lambdas"] = self.lambdas
        state_dict["domain_excess_loss"] = self.domain_excess_loss
        state_dict["seq_len_normalization"] = self.seq_len_normalization
        return state_dict

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.domain_weights = state["domain_weights"]
        self.trajectory_domain_weights = state["trajectory_domain_weights"]
        self.num_updates = state["num_updates"]
        self.lambdas = state["lambdas"]
        self.domain_excess_loss = state["domain_excess_loss"]
        self.seq_len_normalization = state["seq_len_normalization"]

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.BEFORE_FORWARD or event == Event.AFTER_FORWARD or
                event == Event.FIT_END or event == Event.BATCH_END)

    def _load_domain_weights(self, doremi_iter: int) -> torch.Tensor:
        load_dir = re.sub(r"iter-(\d+)", f"iter-{doremi_iter - 1}",
                          self.save_dir)
        weights_path = os.path.join(
            "oci://mosaicml-internal-checkpoints", load_dir, "final",
            "average_domain_weights.npy")  # Hard fixed for now change later
        with tempfile.NamedTemporaryFile() as tmp_file:
            get_file(weights_path, tmp_file.name, overwrite=True)
            domain_weights = torch.from_numpy(np.load(tmp_file.name))
        return domain_weights

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
                if isinstance(callback, RemoteUploaderDownloader)
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
            average_domain_weights = self.trajectory_domain_weights / self.num_updates
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
        self.domain_excess_loss = self.domain_excess_loss.to(device)
        self.seq_len_normalization = self.seq_len_normalization.to(device)

        # Can probably change to be a batch event but think it might get wiped from the state
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
            domain_excess_loss, _, _, seq_len_normalization = state.model.compute_domain_wise_excess_loss(
                state.outputs,
                state.batch,
                self.domain_weights,
                non_zero_excess=True)

            # Can probably reduce on just rank 0 and set weights there
            dist.all_reduce(domain_excess_loss, "sum")
            dist.all_reduce(seq_len_normalization, "sum")

            self.domain_excess_loss += domain_excess_loss.detach()
            self.seq_len_normalization += seq_len_normalization.detach()

            state.batch_set_item(key="domain_weights",
                                 value=self.domain_weights)

        elif event == Event.BATCH_END:
            domain_excess_loss = self.domain_excess_loss
            seq_len_normalization = torch.maximum(
                self.seq_len_normalization,
                torch.ones_like(self.seq_len_normalization
                               ))  # Avoid changing unused domain weights

            lambdas = domain_excess_loss / seq_len_normalization
            lambdas[(lambdas == 0.0)] = self.lambdas[(lambdas == 0.0)]
            self.lambdas = lambdas

            if self.log_excess_loss and dist.get_global_rank() == 0:
                to_log = {}
                for domain_idx, excess_loss in enumerate(lambdas):
                    excess_loss = excess_loss.cpu().item()
                    to_log[
                        f"Excess-loss/domain-{PILE_DATA_SOURCES[domain_idx]}"] = excess_loss

                logger.log_metrics(to_log)

            if int(state.timestamp.batch) < self.warmup_steps:
                state.batch_set_item(key="domain_weights",
                                     value=self.domain_weights)
                self._log_domain_weights(state)
                return

            domain_weights_prime = self.domain_weights * torch.exp(
                self.step_size * lambdas)
            domain_weights_prime = domain_weights_prime / torch.sum(
                domain_weights_prime)  # Normalizing domain weight update

            domain_weights = (
                1 - self.smoothing
            ) * domain_weights_prime + self.smoothing * self.smoothing_dist  # Smooth w/ uniform domain dist

            self.domain_weights = domain_weights
            self.trajectory_domain_weights += domain_weights
            self.num_updates += 1

            # Broadcasting domain weight information
            self._log_domain_weights(state)

            # Resetting trackers (self.lambdas set to be same above)
            self.seq_len_normalization = torch.zeros_like(
                self.seq_len_normalization)
            self.domain_excess_loss = torch.zeros_like(self.domain_excess_loss)

        dist.barrier()
