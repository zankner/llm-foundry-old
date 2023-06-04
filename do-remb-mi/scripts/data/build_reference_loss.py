import os
from typing import Any

import torch
from torch.utils.data import DataLoader, default_collate
from composer import State, Trainer, Callback
from composer.callbacks import RuntimeEstimator
from composer.loggers import Logger
import composer.utils.dist as dist
from streaming import MDSWriter, StreamingDataset

from llmfoundry.data.text_data import ConcatenatedSequenceCollatorWrapper


class ReferenceLossCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        streaming_writer = None
        if dist.get_global_rank() == 0:
            columns = {"tokens": "bytes", "loss": "float32"}
            streaming_writer = MDSWriter(columns=columns,
                                         out=os.path.join(
                                             "/tmp", "pre-concat", "domains"),
                                         compression="ztsd")
        self.streaming_writer = streaming_writer

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        ref_logits = state.outputs.logits
        targets = state.model.get_targets(state.batch)
        b, n_tokens, vocab_size = ref_logits.shape

        losses = state.model.loss_fn(ref_logits.view(-1, vocab_size),
                                     targets.view(-1))
        losses = losses.view(b, n_tokens)

        if dist.get_global_rank() == 0:
            all_losses = torch.vstack(dist.all_gather(losses))
            all_tokens = torch.vstack(dist.all_gather(targets))

            for loss, token in zip(all_losses, all_tokens):
                self.streaming_writer.write({"tokens": token, "ref_loss": loss})
        dist.barrier()

    def eval_end(self, state: State, logger: Logger) -> None:
        self.streaming_writer.finish()


def main(args):
    device_batch_size = args.batch_size / dist.get_world_size()
    assert args.batch_size % device_batch_size != 0, "Batch size must be divisible by the number of devices"

    for domain_id in args.num_domains:
        domain_ds = StreamingDataset(remote=f"oci://",
                                     local=f"/tmp/streaming/domain-{domain_id}",
                                     shuffle=False)
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=default_collate, eos_token_id=0,
            bos_token_id=None)  #Fix
        dataloader = DataLoader(
            domain_ds,
            collate_fn=collate_fn,
            batch_size=device_batch_size,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        model = None

        ref_loss_callback = ReferenceLossCallback()
        runtime_callback = RuntimeEstimator()

        trainer = Trainer(model=model,
                          eval_dataloader=dataloader,
                          progress_bar=True,
                          load_path=args.reference_model_ckpt,
                          callbacks=[ref_loss_callback, runtime_callback])
        trainer.eval()
