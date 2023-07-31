import os
import argparse

from mcli import RunConfig, create_run
from omegaconf import OmegaConf as om

from pretrain_utils import build_model_arch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=str, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ckpt-prefix", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--local-debug", action="store_true")
    parser.add_argument("--eval-type",
                        type=str,
                        required=True,
                        choices=["final", "sweep"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    args = parser.parse_args()

    base_run = RunConfig.from_file(f"rho/yamls/gauntlet.yaml")

    # Set name
    base_run.run_name = f"gauntlet-{args.model_name}"
    base_run.name = f"gauntlet-{args.model_name}"
    base_run.parameters["models"][0]["model_name"] = args.model_name

    # Set compute
    base_run.cluster = args.cluster
    base_run.num_gpus = args.ngpus

    # Set rest of cluster params
    if args.cluster == "r9z1":
        base_run.image = "mosaicml/llm-foundry:2.0.1_cu118-latest"
        base_run.gpu_type = "h100_80gb"
    elif args.cluster in ["r8z6", "r1z1"]:
        base_run.image = "mosaicml/llm-foundry:1.13.1_cu117-latest"
        base_run.gpu_type = "a100_80gb"
    else:
        base_run.image = "mosaicml/llm-foundry:1.13.1_cu117-latest"
        base_run.gpu_type = "a100_40gb"

    # Set batch information
    base_run.parameters["device_eval_batch_size"] = args.batch_size

    def build_ckpt_path(run_name, ckpt_prefix, step):
        if step == "final":
            step_fmt = "latest-rank0.pt.symlink"
        else:
            step_fmt = f"ba{step}/rank0.pt"
        return os.path.join("oci://mosaicml-internal-checkpoints", ckpt_prefix,
                            run_name, "ckpts", step_fmt)

    def build_wandb_logger(run_name, step):
        return {
            "project": "amortized-selection",
            "group": run_name,
            "tags": ["eval", f"step-{step}"]
        }

    def build_model_cfg(run_name, step, model_size, ckpt_prefix):
        # For now assuming tokenization / max length are fixed
        model_arch = build_model_arch(model_size)
        return {
            "model_name": f"{run_name}-step-{step}",
            "tokenizer": {
                "name": "EleutherAI/gpt-neox-20b",
                "kwargs": {
                    "model_max_length": "${max_seq_len}"
                }
            },
            "model": {
                "name": "mpt_causal_lm",
                "init_device": "meta",
                "max_seq_len": "${max_seq_len}",
                "vocab_size": 50432,
                "attn_config": {
                    "alibi": True,
                    "attn_impl": "triton",
                    "clip_qkv": 6,
                    "attn_use_sequence_id": True
                },
                **model_arch
            },
            "loggers": {
                "wandb": build_wandb_logger(run_name, step)
            }
        }

    # TODO: Fix with seeds once we finish re-writing the rest of the code base
    if args.eval_type == "final":
        base_run.parameters["models"] = [
            build_model_cfg(args.run_name, "final", args.ckpt_prefix, seed)
            for seed in args.seeds
        ]
    elif args.eval_type == "sweep":
        assert args.num_tokens is not None
        if args.num_tokens == "26B":
            steps = []
        models = []
        for seed in args.seeds:
            models += [
                build_model_cfg(args.run_name, step, args.ckpt_prefix)
                for step in steps
            ]
        base_run.parameters["models"] = models

    # Set model information
    model_ckpt = os.path.join("oci://mosaicml-internal-checkpoints",
                              args.model_ckpt)
    base_run.command = base_run.command.replace(r"{model_ckpt}", model_ckpt)

    if args.local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(base_run), f=f)
    else:
        run = create_run(base_run)
        print(f"Created run: {run.name}")