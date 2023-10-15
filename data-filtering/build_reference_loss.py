import argparse
import os

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from composer import State, Trainer, Callback
from composer.loggers import Logger
from composer.metrics.nlp import LanguageCrossEntropy
import composer.utils.dist as dist
import transformers
import pandas as pd
import s3fs

from llmfoundry.data.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       StreamingTextDataset)
from llmfoundry.models.mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_tokenizer

from pretrain_utils import CKPT_BASE, build_dataset_base


class ReferenceLossCallback(Callback):

    def __init__(self, ref_losses_path: str) -> None:
        super().__init__()
        data = None
        if dist.get_global_rank() == 0:
            data = {"losses": [], "uid": []}
        self.data = data
        self.ref_losses_path = ref_losses_path
        self.proxy_loss_fn = nn.CrossEntropyLoss(ignore_index=-100,
                                                 reduction="none")

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        ref_logits = state.outputs.logits
        uid = state.batch["uid"]
        targets = state.model.get_targets(state.batch)
        _, seq_len, vocab_size = ref_logits.shape

        state.model.labels = targets  # So we can compute metrics

        losses = self.proxy_loss_fn(ref_logits.view(-1, vocab_size),
                                    targets.view(-1))
        losses = losses.view(-1, seq_len)
        losses = torch.sum(losses, dim=-1)

        all_losses = torch.vstack(dist.all_gather(losses)).view(-1)
        all_uid = torch.vstack(dist.all_gather(uid)).view(-1)
        if dist.get_global_rank() == 0:
            for loss, uid in zip(all_losses, all_uid):
                loss = loss.cpu().item()
                uid = uid.cpu().item()
                self.data["losses"].append(loss)
                self.data["uid"].append(uid)
        dist.barrier()

    def eval_end(self, state: State, logger: Logger) -> None:
        if dist.get_global_rank() == 0:
            data_df = pd.DataFrame(self.data)
            with s3fs.S3FileSystem().open(self.ref_losses_path, 'wb') as f:
                data_df.to_parquet(f, engine="pyarrow")
        dist.barrier()


def build_model(model_size, tokenizer, max_seq_len, vocab_size):
    if model_size == "125M":
        model_cfg = OmegaConf.create({
            "name": "mpt_causal_lm",
            "init_device": "meta",
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "expansion_ratio": 4,
            "max_seq_len": max_seq_len,
            "vocab_size": vocab_size,
            "no_bias": True,
            "norm_type": "low_precision_layernorm",
            "emb_pdrop": 0,
            "resid_pdrop": 0,
            "init_config": {
                "init_nonlinearity": "relu",
                "name": "kaiming_normal_"
            },
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_uses_sequence_id": False,
                "attn_pdrop": 0
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
            "vocab_size": vocab_size,
            "no_bias": True,
            "norm_type": "low_precision_layernorm",
            "emb_pdrop": 0,
            "resid_pdrop": 0,
            "init_config": {
                "init_nonlinearity": "relu",
                "name": "kaiming_normal_"
            },
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_uses_sequence_id": False,
                "attn_pdrop": 0
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
            "vocab_size":
                vocab_size,  # update for hero run with custom tokenizer
            "no_bias": True,
            "norm_type": "low_precision_layernorm",
            "emb_pdrop": 0,
            "resid_pdrop": 0,
            "init_config": {
                "init_nonlinearity": "relu",
                "name": "kaiming_normal_"
            },
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_uses_sequence_id": False,
                "attn_pdrop": 0
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


def build_tokenizer_kwargs(tokenizer_name: str, args):
    if "tiktoken" in tokenizer_name:
        return {"model_name": "gpt-4"}
    elif "gpt-neox-20b" == tokenizer_name:
        return {"model_max_length": args.seq_len}
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def main(args):

    # Building the remote name
    remote_download = build_dataset_base(args.dataset, args.tokenizer,
                                         args.seq_len, args.final_num_tokens,
                                         args.num_passes, False)
    reference_run_data_suffix = f"{args.final_num_tokens}-tokens-from-{args.num_passes}-passes-ref-{args.ref_model_size}-{args.ref_num_tokens}-sd-{args.train_seed}"
    remote_upload = os.path.join(
        *remote_download.replace("s3://", "").split("/")[:-3],
        reference_run_data_suffix, "heuristic.parquet")
    print(f"Uploading losses to: {remote_upload}")

    global_batch_size = args.device_batch_size * dist.get_world_size()
    print(f"Global batch size: {global_batch_size}")

    # Building model ckpt name
    ref_run_name = f"{args.dataset}-passes-{args.num_passes}-ref-{args.ref_model_size}-{args.ref_num_tokens}-sd-{args.train_seed}"
    ref_run_name = f"{args.tokenizer}-{args.global_batch_size}-{args.seq_len}-{ref_run_name}"
    ref_ckpt = os.path.join(CKPT_BASE, args.dataset, "reference", ref_run_name,
                            "ckpts", "latest-rank0.pt.symlink")

    if args.tokenizer == "gpt4-tiktoken":
        tokenizer_name = "tiktoken"
        vocab_size = 100352
    elif args.tokenizer == "gpt-neox-20b":
        tokenizer_name = "EleutherAI/gpt-neox-20b"
        vocab_size = 50432
    else:
        raise ValueError(f"Unknown tokenizer: {args.tokenizer}")
    tokenizer = build_tokenizer(tokenizer_name,
                                tokenizer_kwargs=build_tokenizer_kwargs(
                                    args.tokenizer, args))
    model, fsdp_cfg = build_model(args.ref_model_size,
                                  tokenizer=tokenizer,
                                  max_seq_len=args.seq_len,
                                  vocab_size=vocab_size)
    model.use_logits = False  # So that full ModellingOutputs passed to callback
    model.val_metrics = {
        LanguageCrossEntropy.__name__: LanguageCrossEntropy()
    }  # Only metric we care about

    print(f"Downloading data from {remote_download}")
    dataset = StreamingTextDataset(
        remote=remote_download,
        tokenizer=tokenizer,
        max_seq_len=args.seq_len,
        batch_size=args.device_batch_size,
        local="/tmp/train-data",
        split=None,
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

    reference_loss_writer = ReferenceLossCallback(ref_losses_path=remote_upload)
    trainer = Trainer(
        model=model,
        eval_dataloader=dataloader,
        callbacks=reference_loss_writer,
        progress_bar=True,
        load_path=ref_ckpt,
        fsdp_config=fsdp_cfg,
    )

    trainer.eval()
    trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Ref model args
    parser.add_argument("--ref-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])
    parser.add_argument("--train-seed", type=int, required=True)

    # Final model args
    parser.add_argument("--final-num-tokens",
                        type=str,
                        required=True,
                        choices=["26B", "52B", "DEBUG"])

    # Data args
    parser.add_argument("--device-batch-size", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="gpt4-tiktoken")
    parser.add_argument("--eos-token-id", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-passes", type=str, required=True)
    parser.add_argument("--dataset",
                        type=str,
                        default="mpt",
                        choices=["mpt", "pile"])

    args = parser.parse_args()

    dist.initialize_dist(
        device="gpu", timeout=1200)  # Setting high dist timeout for slow upload

    main(args)
