import os
import argparse

from mcli import RunConfig, create_run
from omegaconf import OmegaConf as om

from pretrain_utils import (CKPT_BASE, build_model_arch, build_final_base,
                            build_proxy_base)


def build_ckpt_path(run_name, run_type, step, dataset, seed):
    if step == "final":
        step_fmt = "latest-rank0.pt.symlink"
    else:
        step_fmt = f"ba{step}/rank0.pt"
    return os.path.join(CKPT_BASE, dataset, run_type, f"{run_name}-sd-{seed}",
                        "ckpts", step_fmt)


def build_wandb_logger(run_name, step, model_tags):
    return {
        "project": "amortized-selection",
        "group": f"eval-{run_name}",
        "tags": model_tags + [f"step-{step}"]
    }


def build_model_cfg(run_name, step, model_size, run_type, model_tags, dataset,
                    seed):
    # For now assuming tokenization / max length are fixed
    model_arch = build_model_arch(model_size)
    return {
        "run_name": f"eval-{run_name}-step-{step}-sd-{seed}",
        "model_name": f"eval-{run_name}-step-{step}",
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
            "expansion_ratio": 4,
            "no_bias": True,
            "attn_config": {
                "alibi": True,
                "attn_impl": "triton",
                "clip_qkv": 6,
                "attn_use_sequence_id": True
            },
            **model_arch
        },
        "loggers": {
            "wandb": build_wandb_logger(run_name, step, model_tags)
        },
        "load_path": build_ckpt_path(run_name, run_type, step, dataset, seed)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=str, default=8)
    parser.add_argument("--device-batch-size", type=int, default=8)

    # Run config
    parser.add_argument("--run-type",
                        type=str,
                        required=True,
                        choices=["final", "reference", "proxy"])
    parser.add_argument("--eval-type",
                        type=str,
                        required=True,
                        choices=["final", "sweep"])
    parser.add_argument(
        "--selection-algo",
        type=str,
        required=True,
        choices=["baseline", "rho", "hard-mine", "easy-mine",
                 "baseline"])  # Treat baseline as a selection algo

    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])
    parser.add_argument(
        "--full-batch-size",
        help="Batch size for points to be labeled that will then be pruned",
        type=int,
        choices=[1024, 2048, 4096])
    parser.add_argument("--num-pplx-filter", type=int, default=0)

    # Final args
    parser.add_argument("--final-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--final-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])

    # Misc
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])
    parser.add_argument("--dataset", type=str, default="pile", required=True)
    parser.add_argument("--local-debug", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    base_run = RunConfig.from_file(f"amortized-obs/yamls/gauntlet.yaml")

    # Building the run name
    if args.run_type == "final":
        model_size = args.final_model_size
        num_tokens = args.final_num_tokens
        model_tags = [
            f"model-size-{args.final_model_size}",
            f"num-tokens-{args.final_num_tokens}"
        ]
        final_base = build_final_base(args.final_num_tokens,
                                      args.final_model_size)
        if args.selection_algo == "baseline":
            run_name = f"final-{args.dataset}-baseline-{final_base}-holdt-{args.holdout_num_tokens}"
        else:
            proxy_run_base = build_proxy_base(
                args.selection_algo, args.proxy_num_tokens,
                args.proxy_model_size, args.full_batch_size,
                args.num_pplx_filter, args.ref_num_tokens, args.ref_model_size)
            fmt_proxy_run_base = proxy_run_base.replace(
                f"{args.selection_algo}-", "")
            run_name = f"final-{args.dataset}-{args.selection_algo}-{final_base}-{fmt_proxy_run_base}-holdt-{args.holdout_num_tokens}"
    elif args.run_type == "proxy":
        model_tags = [
            f"model-size-{args.proxy_model_size}",
            f"num-tokens-{args.proxy_num_tokens}"
        ]
        model_size = args.proxy_model_size
        num_tokens = args.proxy_num_tokens
        proxy_run_base = build_proxy_base(
            args.selection_algo, args.proxy_num_tokens, args.proxy_model_size,
            args.full_batch_size, args.num_pplx_filter, args.ref_num_tokens,
            args.ref_model_size)
        run_name = f"proxy-{args.dataset}-{proxy_run_base}-holdt-{args.holdout_num_tokens}"
    else:
        raise ValueError("Not supporting eval reference runs yet")

    # Just setting a whole bunch of tags
    model_tags += [
        f"holdt-{args.holdout_num_tokens}",
        f"refp-{args.ref_model_size}",
        f"reft-{args.ref_num_tokens}",
        f"proxp-{args.proxy_model_size}",
        f"proxt-{args.proxy_num_tokens}",
        f"fb-{args.full_batch_size}",
        f"fillpplx-{args.num_pplx_filter}",
        f"fp-{args.final_model_size}",
        f"ft-{args.final_num_tokens}",
        args.selection_algo,
        "eval",
        f"seed-{args.seed}",
        args.run_type,
    ]
    # Set name
    base_run.run_name = f"eval-{run_name}"
    base_run.name = f"eval-{run_name}"

    # Set seed for reasons
    base_run.parameters["seed"] = args.seed

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
    base_run.parameters["device_eval_batch_size"] = args.device_batch_size

    if args.eval_type == "final":
        base_run.parameters["models"] = [
            build_model_cfg(run_name, "final", model_size, args.run_type,
                            model_tags, args.dataset, args.seed)
        ]
    elif args.eval_type == "sweep":
        assert args.num_tokens is not None

        if num_tokens == "26B":
            start_step = 1_000
            end_step = 24_000
        else:
            raise ValueError("need to define steps for given token count")

        # Fixing to be eval every 1k batches for now
        steps = list(range(start_step, end_step + 1, 1_000))

        models = [
            build_model_cfg(run_name, step, model_size, args.run_type,
                            model_tags, args.dataset, args.seed)
            for step in steps
        ]
        base_run.parameters["models"] = models

    if args.local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(base_run), f=f)
    else:
        run = create_run(base_run)
        print(f"Created run: {run.name}")