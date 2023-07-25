import argparse

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from composer import State, Trainer, Callback
from composer.loggers import Logger, WandBLogger
from composer.metrics.nlp import LanguageCrossEntropy
import composer.utils.dist as dist
from streaming import MDSWriter
import transformers
from transformers import AutoTokenizer

from llmfoundry.data.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       StreamingTextDataset)
from llmfoundry.models.mpt import ComposerMPTCausalLM


class ReferenceLossCallback(Callback):

    def __init__(self, streaming_writer_path: str) -> None:
        super().__init__()
        streaming_writer = None
        if dist.get_global_rank() == 0:
            columns = {
                "tokens": "bytes",
                "ref_losses": "bytes",
                "idx": "int"
            }
            streaming_writer = MDSWriter(columns=columns,
                                         out=streaming_writer_path,
                                         compression="zstd")
        self.streaming_writer = streaming_writer
        self.proxy_loss_fn = nn.CrossEntropyLoss(ignore_index=-100,
                                                 reduction="none")

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        ref_logits = state.outputs.logits
        tokens = state.batch["input_ids"]
        idx = state.batch["idx"]
        targets = state.model.get_targets(state.batch)
        _, _, vocab_size = ref_logits.shape

        state.model.labels = targets  # So we can compute metrics

        losses = self.proxy_loss_fn(ref_logits.view(-1, vocab_size),
                                    targets.view(-1))
        losses = losses.view(-1)

        all_losses = torch.vstack(dist.all_gather(losses))
        all_tokens = torch.vstack(dist.all_gather(tokens))
        all_idx = torch.vstack(dist.all_gather(idx))
        if dist.get_global_rank() == 0:
            for losses, tokens, uids in zip(all_losses, all_tokens, all_idx):
                byte_losses = losses.cpu().numpy().tobytes()
                byte_tokens = tokens.cpu().numpy().tobytes()
                ind_idx = uids.cpu().item()
                self.streaming_writer.write({
                    "tokens": byte_tokens,
                    "ref_losses": byte_losses,
                    "idx": ind_idx
                })
        dist.barrier()

    def eval_end(self, state: State, logger: Logger) -> None:
        if dist.get_global_rank() == 0:
            self.streaming_writer.finish()
        dist.barrier()


def build_model(model_size, tokenizer, max_seq_len):
    if model_size == "125M":
        model_cfg = OmegaConf.create({
            "name": "mpt_causal_lm",
            "init_device": "meta",
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "expansion_ratio": 4,
            "max_seq_len": max_seq_len,
            "vocab_size": 50432,
            "no_bias": True,
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_uses_sequence_id": True
            }
        })
        fsdp_cfg = OmegaConf.create({
            "activation_checkpointing": False,
            "activation_checkpointing_reentrant": False,
            "activation_cpu_offload": False,
            "limit_all_gathers": True,
            "mixed_precision": "PURE",
            "sharding_strategy": "FULL_SHARD",
            "state_dict_type": "full",
            "verbose": False
        })
        return ComposerMPTCausalLM(model_cfg, tokenizer), fsdp_cfg
    elif model_size == "250M":
        model_cfg = OmegaConf.create({
            "name": "mpt_causal_lm",
            "init_device": "meta",
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 16,
            "expansion_ratio": 4,
            "max_seq_len": max_seq_len,
            "vocab_size": 50432,
            "no_bias": True,
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_uses_sequence_id": True
            }
        })
        fsdp_cfg = OmegaConf.create({
            "activation_checkpointing": False,
            "activation_checkpointing_reentrant": False,
            "activation_cpu_offload": False,
            "limit_all_gathers": True,
            "mixed_precision": "PURE",
            "sharding_strategy": "FULL_SHARD",
            "state_dict_type": "full",
            "verbose": False
        })
        return ComposerMPTCausalLM(model_cfg, tokenizer), fsdp_cfg
    elif model_size == "1B":
        model_cfg = OmegaConf.create({
            "name": "mpt_causal_lm",
            "init_device": "meta",
            "d_model": 2048,
            "n_heads": 16,
            "n_layers": 24,
            "expansion_ratio": 4,
            "max_seq_len": 2048,
            "vocab_size": 50432,  # update for hero run with custom tokenizer
            "no_bias": True,
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_uses_sequence_id": True
            }
        })
        fsdp_cfg = OmegaConf.create({
            "activation_checkpointing": False,
            "activation_checkpointing_reentrant": False,
            "activation_cpu_offload": False,
            "limit_all_gathers": True,
            "mixed_precision": "PURE",
            "sharding_strategy": "FULL_SHARD",
            "state_dict_type": "full",
            "verbose": False
        })
        return ComposerMPTCausalLM(model_cfg, tokenizer), fsdp_cfg
    else:
        raise ValueError(f"Unkown model size: {model_size}")


def main(args):
    global_batch_size = args.device_batch_size * dist.get_world_size()
    print(f"Global batch size: {global_batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model, fsdp_cfg = build_model(
        args.model_size, tokenizer=tokenizer, max_seq_len=args.max_seq_len
    )
    model.use_logits = False  # So that full ModellingOutputs passed to callback
    model.val_metrics = {
        LanguageCrossEntropy.__name__: LanguageCrossEntropy()
    }  # Only metric we care about

    loggers = []
    if not args.no_wandb:
        loggers = [
            WandBLogger(project="amortize-selection-preprocess",
                        entity="mosaic-ml",
                        name=f"{args.model_size}-ref-loss")
        ]

    dataset = StreamingTextDataset(
        remote=args.download_remote,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.device_batch_size,
        remote=args.data_remote,
        local=args.local,
        split="train"
        shuffle=False,
    )
    lm_collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer, mlm=False)
    collate_fn = ConcatenatedSequenceCollatorWrapper(
        base_collator=lm_collate_fn,
        eos_token_id=args.eos_token_id,
        bos_token_id=None)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.device_batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    reference_loss_writer = ReferenceLossCallback(
        streaming_writer_path=args.upload_remote)
    trainer = Trainer(model=model,
                      eval_dataloader=dataloader,
                      callbacks=reference_loss_writer,
                      progress_bar=True,
                      load_path=args.model_ckpt,
                      fsdp_config=fsdp_cfg,
                      loggers=loggers)

    trainer.eval()
    trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-batch-size", type=int, default=256)
    parser.add_argument("--download-remote", type=str, required=True)
    parser.add_argument("--upload-remote", type=str, required=True)
    parser.add_argument("--model-size",
                        type=str,
                        choices=["125M", "250M", "1B"],
                        required=True)
    parser.add_argument("--model-ckpt", type=str, required=True)
    parser.add_argument("--tokenizer",
                        type=str,
                        default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--eos-token-id", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    dist.initialize_dist(device="gpu")

    main(args)