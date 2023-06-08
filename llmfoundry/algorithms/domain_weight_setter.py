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


class DomainWeightSetter(Algorithm):

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
        self.log_domain_weights_freq = log_domain_weights_freq

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.BEFORE_TRAIN_BATCH or
                event == Event.AFTER_FORWARD or event == Event.FIT_END or
                event == Event.AFTER_TRAIN_BATCH)

    def _upload_data(self, data, path: str, uploader: RemoteUploaderDownloader,
                     state: State):
        #print(data)
        with tempfile.NamedTemporaryFile() as tmp_file:
            np.save(tmp_file.name + ".npy", data)
            #print(np.load(tmp_file.name + ".npy"))
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
            average_domain_weights = self.trajectory_domain_weights / (
                int(state.timestamp.batch) + 1)
            self._upload_data(average_domain_weights.cpu().numpy(),
                              path=average_domain_weights_path,
                              uploader=remote_uploader,
                              state=state)

    @torch.no_grad()
    def apply(self, event: Event, state: State, logger: Logger) -> None:
        device = state.batch["input_ids"].device
        self.domain_weights = self.domain_weights.to(device)
        self.trajectory_domain_weights = self.trajectory_domain_weights.to(
            device)
        state.batch_set_item(key="domain_weights", value=self.domain_weights)

        if event == Event.BEFORE_TRAIN_BATCH:
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
            domain_excess_loss, seq_len_normalization = state.model.compute_domain_wise_excess_loss(
                state.outputs,
                state.batch,
                self.domain_weights,
                non_zero_excess=True)

            dist.all_reduce(domain_excess_loss, "sum")
            dist.all_reduce(seq_len_normalization, "sum")

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

        dist.barrier()
