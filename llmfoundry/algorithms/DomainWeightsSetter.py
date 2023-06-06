# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the domain weights."""
import tempfile
import os

import torch
import numpy as np
from composer import Algorithm, Event, State
from composer.utils import dist
from composer.loggers import Logger, RemoteUploadedDownloader


class DomeainWeightSetter(Algorithm):

    def __init__(
        self,
        num_domains,
        save_dir,
        step_size=1.0,
        smoothing=1e-4,
        init_dist="uniform",
        log_domain_weights_freq=100,
    ):

        self.step_size = step_size
        self.smoothing = smoothing

        if init_dist == "uniform":
            self.domain_weights = torch.ones(num_domains) / num_domains
        else:
            # self.domain_weights = init_dist
            raise NotImplementedError(
                "Only uniform initialization is supported")

        self.trajectory_domain_weights = self.domain_weights

        self.save_dir = save_dir
        self._log_domain_weights_freq = log_domain_weights_freq

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_FORWARD or event == Event.INIT or event == Event.FIT_END

    def _upload_data(self, data, path: str, uploader: RemoteUploadedDownloader,
                     state: State):
        with tempfile.NamedTemporaryFile() as tmp_file:
            np.save(tmp_file, data)
            uploader.upload_file(state,
                                 remote_file_name=os.path.join(
                                     self.save_dir, path),
                                 local_file_name=tmp_file.name)

    def _log_domain_weights(self, state: State, final: bool = False) -> None:
        remote_uploader = [
            logger for logger in state.loggers
            if isinstance(logger, RemoteUploadedDownloader)
        ]
        assert len(
            remote_uploader) == 1, "Only one remote uploader is supported"
        remote_uploader = remote_uploader[0]

        if (
                state.timestamp.batch_in_epoch + 1
        ) % self.log_domain_weights_freq != 0 and state.timestamp.batch != 0 and not final:
            return

        if final:
            prefix = "final"
        else:
            prefix = f"epoch-{state.timestamp.epoch}-ba-{state.timestamp.batch_in_epoch}"
        domain_weights_path = os.path.join(prefix, "domain_weights.npy")
        average_domain_weights_path = os.path.join(
            prefix, "average_domain_weights.npy")

        self._upload_data(self.domain_weights.cpu().numpy(),
                          path=domain_weights_path,
                          uploader=remote_uploader,
                          state=state)
        average_domain_weights = self.trajectory_domain_weights / (
            state.timestamp.batch + 1)
        self._upload_data(average_domain_weights.cpu().numpy(),
                          path=average_domain_weights_path,
                          uploader=remote_uploader,
                          state=state)

    @torch.no_grad()
    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.INIT:
            self._log_domain_weights(state)
        elif event == Event.FIT_END:
            self._log_domain_weights(state, final=True)
        elif event == Event.AFTER_FORWARD:
            domain_excess_loss, seq_len_normalization = self.model.compute_domain_wise_excess_loss(
                state.outputs,
                state.batch,
                self.domain_weights,
                non_zero_excess=True)

            domain_excess_loss = dist.all_reduce(domain_excess_loss, "sum")
            seq_len_normalization = dist.all_reduce(seq_len_normalization,
                                                    "sum")

            seq_len_normalization = torch.maximum(
                seq_len_normalization, torch.ones_like(seq_len_normalization)
            )  # Avoid changing unused domain weights

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

            state.batch_set_item(key="domain_weights",
                                 value=self.domain_weights)

            self._log_domain_weights(state)