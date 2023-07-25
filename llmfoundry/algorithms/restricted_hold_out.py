from composer import Algorithm, Event, State, Logger
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
        excess_loss = proxy_losses - microbatch[""]

    def apply(self, event: Event, state: State, logger: Logger):
        if event != Event.AFTER_DATALOADER:
            return

        full_batch = state.batch
        microbatches = self._train_data_spec.split_batch(
            full_batch, self.state.device_train_microbatch_size)
        for microbatch in microbatches:
            ...