from typing import Any, Dict, List

from composer import Algorithm, Event, State, Logger
from composer.core import data_spec, get_precision_context
from composer.utils import dist
import torch
import torch.distributed as torch_dist
from torch.nn import CrossEntropyLoss


def dist_scatter(tensor: torch.Tensor, scatter_list: List[torch.Tensor],
                 src: int) -> None:
    """scatter a tensor list the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes participating in the collective.
    See :func:`torch.distributed.broadcast`.

    Args:
        tensor (torch.Tensor): Data to be sent if ``src`` is the rank of current process,
            and tensor to be used to save received data otherwise.
        src (int): Source rank
    """
    if dist.is_available() and dist.is_initialized():
        if dist.get_global_rank() == src:
            torch_dist.scatter(tensor, scatter_list=scatter_list, src=src)
        else:
            torch_dist.scatter(tensor, scatter_list=[], src=src)
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    raise RuntimeError(
        f'The world_size({world_size}) > 1, but the distributed package is not '
        'available or has not been initialized. Please check you have initialized '
        'the distributed runtime and that PyTorch has been built with distributed '
        'support. If calling this function outside Trainer, please ensure that '
        '`composer.utils.dist.initialize_dist` has been called first.')


class RestrictedHoldOut(Algorithm):

    def __init__(self, num_subsample: int):
        self.num_subsample = num_subsample
        assert self.num_subsample % dist.get_world_size(
        ) == 0, "num_subsample must be divisible by world size"  # Should really check world_size * micro_batch_size
        self.device_num_subsample_size = self.num_subsample // dist.get_world_size(
        )

        self.loss_fn = CrossEntropyLoss(reduction="none", ignore_index=-100)

        self.data_trajectory = []

    def match(self, event: Event, state: State):
        return event == Event.AFTER_DATALOADER

    @torch.no_grad()
    def _compute_excess_loss(self, microbatch, model, state):
        with torch.no_grad(),\
                get_precision_context(state.precision):
            b, seq_len = microbatch["input_ids"].shape
            targets = model.get_targets(microbatch)
            proxy_outputs = model(microbatch)
            proxy_losses = self.loss_fn(
                proxy_outputs.logits.view(-1, proxy_outputs.logits.size(-1)),
                targets.view(-1))
            proxy_losses = proxy_losses.view(b, seq_len).sum(dim=-1)
            excess_loss = proxy_losses - microbatch["ref_loss"]
        return excess_loss, proxy_losses

    # Maybe set model in eval mode
    def apply(self, event: Event, state: State, logger: Logger):
        if event != Event.AFTER_DATALOADER:
            return

        full_batch = state.batch
        microbatches = data_spec._default_split_batch(
            full_batch, state.device_train_microbatch_size)

        excess_loss = []
        proxy_loss = []
        for microbatch in microbatches:
            micro_excess_loss, micro_proxy_loss = self._compute_excess_loss(
                microbatch, state.model, state)
            excess_loss.append(micro_excess_loss)
            proxy_loss.append(micro_proxy_loss)
        excess_loss = torch.cat(excess_loss).view(-1)
        proxy_loss = torch.cat(proxy_loss).view(-1)

        gathered_batch = {}
        for k, v in full_batch.items():
            gathered_batch[k] = torch.cat(dist.all_gather(v))
        gathered_excess = torch.cat(dist.all_gather(excess_loss))
        gathered_proxy = torch.cat(dist.all_gather(proxy_loss))

        if dist.get_global_rank() == 0:
            # Select points with highest excess loss
            sorted_idx = torch.argsort(gathered_excess, descending=True)
            subsample_idx = sorted_idx[:self.num_subsample]
            skipsample_idx = sorted_idx[self.num_subsample:]

            # Subsample the batch based on selected indices
            subsampled_batch = {}
            for k, v in gathered_batch.items():
                subsampled_batch[k] = v[subsample_idx]

            # Logging losses
            to_log = {}
            _, seq_len = gathered_batch["input_ids"].shape # Logging mean over tokens

            excess_loss_selected = gathered_excess[subsample_idx].mean().cpu(
            ).item() / seq_len
            excess_loss_leftout = gathered_excess[skipsample_idx].mean().cpu(
            ).item() / seq_len
            to_log["proxy/excess-loss/selected"] = excess_loss_selected
            to_log["proxy/excess-loss/leftout"] = excess_loss_leftout

            ref_loss_selected = gathered_batch["ref_loss"][subsample_idx].mean(
            ).cpu().item() / seq_len
            ref_loss_leftout = gathered_batch["ref_loss"][skipsample_idx].mean(
            ).cpu().item() / seq_len
            to_log["proxy/ref-loss/selected"] = ref_loss_selected
            to_log["proxy/ref-loss/leftout"] = ref_loss_leftout

            proxy_loss_selected = gathered_proxy[subsample_idx].mean().cpu(
            ).item() / seq_len
            proxy_loss_leftout = gathered_proxy[skipsample_idx].mean().cpu(
            ).item() / seq_len
            to_log["proxy/proxy-loss/selected"] = proxy_loss_selected
            to_log["proxy/proxy-loss/leftout"] = proxy_loss_leftout

            logger.log_metrics(to_log)

            # Write the selected ds indices to state
            subsample_og_idx = gathered_batch["idx"][subsample_idx]
            self.data_trajectory.append(subsample_og_idx.cpu().tolist())
        else:
            subsampled_batch = {}
            for k, v in gathered_batch.items():
                subsampled_batch[k] = torch.zeros_like(v)[:self.num_subsample]
        dist.barrier()

        for k, v in subsampled_batch.items():
            device_v = torch.zeros_like(v)[:self.device_num_subsample_size]
            dist_scatter(tensor=device_v,
                         scatter_list=list(v.chunk(dist.get_world_size())),
                         src=0)
            dist.barrier()  # Can probably get rid of need to check
            state.batch_set_item(k, device_v)

        dist.barrier()

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["data_trajectory"] = self.data_trajectory
        return state_dict

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.data_trajectory = state["data_trajectory"]