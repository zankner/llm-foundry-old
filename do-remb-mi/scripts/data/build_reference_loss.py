import argparse

import torch
from torch.utils.data import DataLoader, default_collate
from composer import State, Trainer, Callback
from composer.callbacks import RuntimeEstimator
from composer.loggers import Logger
import composer.utils.dist as dist
from streaming import MDSWriter
from transformers import AutoTokenizer

from llmfoundry.data.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       StreamingTextDataset)


class ReferenceLossCallback(Callback):

    def __init__(self, writer_path: str) -> None:
        super().__init__()
        streaming_writer = None
        if dist.get_global_rank() == 0:
            columns = {"tokens": "bytes", "loss": "float32"}
            streaming_writer = MDSWriter(columns=columns,
                                         out=writer_path,
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

            for losses, tokens in zip(all_losses, all_tokens):
                byte_losses = losses.numpy().tobytes()
                byte_tokens = tokens.numpy().tobytes()
                self.streaming_writer.write({
                    "tokens": byte_tokens,
                    "ref_loss": byte_losses
                })
        dist.barrier()

    def eval_end(self, state: State, logger: Logger) -> None:
        self.streaming_writer.finish()


def main(args):
    device_batch_size = args.batch_size / dist.get_world_size()
    assert args.batch_size % device_batch_size != 0, "Batch size must be divisible by the number of devices"

    for domain_id in args.num_domains:
        print(f"Starting to build losses for domain {domain_id}...")

        streaming_writer_path = f"/tmp/domains/domain-{domain_id}/train"

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dataset = StreamingTextDataset(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            remote=
            f"oci://mosaicml-internal-doremi/pile/pre-concat/gpt-neox-20b-seqlen-2048/data-sources/domain-{domain_id}",
            local=f"/tmp/streaming/domain-{domain_id}",
            split="train",
            batch_size=device_batch_size,
            shuffle=False,
            shuffle_algo='py1b',
        )
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=default_collate,
            eos_token_id=args.eos_token_id,
            bos_token_id=None)  #Fix
        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=device_batch_size,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        model = None

        ref_loss_callback = ReferenceLossCallback(
            writer_path=streaming_writer_path)
        runtime_callback = RuntimeEstimator()

        trainer = Trainer(model=model,
                          eval_dataloader=dataloader,
                          progress_bar=True,
                          load_path=args.reference_model_ckpt,
                          callbacks=[ref_loss_callback, runtime_callback])
        trainer.eval()
        trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--reference-model-ckpt", type=str, required=True)
    parser.add_argument("--tokenizer",
                        type=str,
                        default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--eos-token-id", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    dist.initialize_dist(device="gpu")

    main(args)
