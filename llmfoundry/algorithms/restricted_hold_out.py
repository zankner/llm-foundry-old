from composer import Algorithm, Event, State, Logger
from composer.utils import dist
import torch
from torch.nn import CrossEntropyLoss


class RestrictedHoldOut(Algorithm):

    def __init__(self, num_subsample: int):
        self.num_subsample = num_subsample
        self.loss_fn = CrossEntropyLoss(reduction="none")

    def match(self, event: Event):
        ...

    @torch.no_grad()
    def _compute_excess_loss(self, microbatch, model):
        b, seq_len = microbatch["input_ids"].shape
        targets = self.get_targets(microbatch)
        proxy_outputs = model(microbatch)
        proxy_losses = self.loss_fn(
            proxy_outputs.logits.view(-1, proxy_outputs.logits.size(-1)),
            targets.view(-1))
        proxy_losses = proxy_losses.view(b, seq_len).sum(dim=-1)
        excess_loss = proxy_losses - ref_losses
        return excess_loss

    # Maybe set model in eval mode
    def apply(self, event: Event, state: State, logger: Logger):
        if event != Event.AFTER_DATALOADER:
            return

        full_batch = state.batch
        microbatches = self._train_data_spec.split_batch(
            full_batch, self.state.device_train_microbatch_size)

        excess_loss = []
        for microbatch in microbatches:
            micro_excess_loss = self._compute_excess_loss(
                microbatch, state.model)
            excess_loss.append(micro_excess_loss)
        excess_loss = torch.cat(excess_loss).view(-1)

        if dist.get_global_rank() == 0:
            # Select points with highest excess loss
            excess_loss = torch.cat(dist.all_gather(excess_loss))
            sorted_idx = torch.argsort(excess_loss)
            subsample_idx = sorted_idx[:self.num_subsample]

            # Subsample the batch based on selected indices
            subsampled_batch = {}
            for k, v in full_batch.items():
                gathered_v = torch.cat(dist.all_gather(v))
                subsampled_batch[k] = gathered_v[subsample_idx]

            # Write the selected ds indices to state
            # TODO

        dist.barrier()
