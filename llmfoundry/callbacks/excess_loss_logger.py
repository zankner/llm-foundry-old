# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the domain wise excess loss"""
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


class ExcessLossLogger(Algorithm):

    def __init__(self, num_domains, save_dir, log_domain_excess_loss_freq=100):

        self.num_domains = num_domains
        self.save_dir = save_dir

        self.excess_loss = torch.zeros(num_domains)
        self.num_updates = torch.ones(num_domains)

        self.log_domain_excess_loss_freq = log_domain_excess_loss_freq

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["excess_loss"] = self.excess_loss
        state_dict["num_updates"] = self.num_updates
        return state_dict

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.excess_loss = state["excess_loss"]
        self.num_updates = state["num_updates"]

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.BEFORE_FORWARD or event == Event.AFTER_FORWARD or
                event == Event.FIT_END or event == Event.BATCH_END)

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

    def _log_domain_excess_loss(self,
                                state: State,
                                final: bool = False) -> None:
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

            if self.log_excess_loss and dist.get_global_rank(
            ) == 0 and self.num_domains == 22:
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
