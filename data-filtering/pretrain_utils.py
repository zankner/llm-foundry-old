import os

from typing import Optional

from omegaconf import OmegaConf as om
from mcli import create_run

CKPT_BASE = "oci://mosaicml-internal-checkpoints/zack/data-filtering/"


def set_common_args(args,
                    base_run,
                    run_name,
                    save_base,
                    data_remote,
                    model_size,
                    duration,
                    seed,
                    save_suffix="",
                    token_multiplier=1):
    # Set run name
    base_run.name = f"sd-{seed}-{run_name.lower()}"[:56]  # Mcli things
    base_run.parameters["run_name"] = run_name

    # Set seed
    if args.overwrite_shuffle_seed is not None:
        base_run.parameters["global_seed"] = args.overwrite_shuffle_seed
    else:
        base_run.parameters["global_seed"] = seed

    # Set compute
    base_run.cluster = args.cluster
    base_run.gpu_num = args.ngpus
    # Set rest of cluster params
    if args.cluster in ["r9z1", "r14z3"]:
        base_run.gpu_type = "h100_80gb"
    elif args.cluster in ["r4z5", "r4z7", "r4z6", "r8z6", "r1z1", "r4z8"]:
        base_run.gpu_type = "a100_80gb"
    else:
        base_run.gpu_type = "a100_40gb"

    # Special images
    if args.cluster in ["r4z5", "r4z7", "r4z6", "r4z8"]:
        base_run.image = "mosaicml/pytorch:2.1.0_cu121-python3.10-ubuntu20.04-aws"
        base_run.env_variables += [{
            "key": "FI_EFA_FORK_SAFE",
            "value": "1"
        }, {
            "key": "NCCL_DEBUG",
            "value": "INFO"
        }]

    # # Set timeout for slow clusters
    # if args.cluster in ["r9z1"]:
    base_run.parameters["dist_timeout"] = 1800

    # Set modeling args
    model_cfg = build_model_arch(model_size)
    lr = model_cfg["lr"] if args.lr is None else args.lr
    base_run.parameters["model"]["d_model"] = model_cfg["d_model"]
    base_run.parameters["model"]["n_heads"] = model_cfg["n_heads"]
    base_run.parameters["model"]["n_layers"] = model_cfg["n_layers"]
    base_run.parameters["optimizer"]["lr"] = lr
    base_run.parameters["optimizer"]["weight_decay"] = lr  # Forcing lr = wd

    # Set ckpt save folder
    save_folder = os.path.join(
        save_base,
        f"{run_name}{'-' + save_suffix if save_suffix else ''}-bs-{args.global_batch_size}-lr-{lr}-sd-{seed}",
        "ckpts")
    base_run.parameters["save_folder"] = save_folder

    # Set data args
    base_run.parameters["train_loader"]["dataset"]["remote"] = data_remote

    # Common wandb tags
    base_run.parameters["loggers"]["wandb"]["tags"] += [
        f"dataset-{args.dataset}",
        f"seed-{seed}",
        f"lr-{lr}",
    ]
    base_run.parameters["loggers"]["wandb"]["name"] = f"{run_name}-sd-{seed}"
    base_run.parameters["loggers"]["wandb"]["group"] = run_name

    # Handle preemption
    if args.preemptible:
        if not args.autoresume:
            print("WARNING: Preemptible training suggested autoresume")
        base_run.scheduling = {"resumable": True, "priority": args.priority}
        base_run.parameters["autoresume"] = True
    else:
        base_run.parameters["autoresume"] = args.autoresume

    # Set batch size
    base_run.parameters["device_train_microbatch_size"] = args.device_batch_size
    base_run.parameters["device_eval_batch_size"] = args.device_batch_size

    # Set predownload based on device batch size
    base_run.parameters["train_loader"]["dataset"]["predownload"] = int(
        args.global_batch_size / args.ngpus) * 4

    # Set training duration
    if duration == "2B":
        duration_tokens = 2_000_000_000
    elif duration == "5B":
        duration_tokens = 5_000_000_000
    elif duration == "20B":
        duration_tokens = 20_000_000_000
    elif duration == "26B":
        duration_tokens = 26_000_000_000
    elif duration == "52B":
        duration_tokens = 52_000_000_000
    elif duration == "130B":
        duration_tokens = 130_000_000_000
    base_run.parameters["max_duration"] = f"{duration_tokens}tok"

    # Set batch information
    base_run.parameters["global_train_batch_size"] = args.global_batch_size

    # Set tokenization args
    base_run.parameters["max_seq_len"] = args.seq_len
    if args.tokenizer == "gpt4-tiktoken":
        base_run.parameters["model"]["vocab_size"] = 100352
        base_run.parameters["tokenizer_name"] = "tiktoken"
        base_run.parameters["tokenizer"]["kwargs"] = {"model_name": "gpt-4"}
    elif args.tokenizer == "gpt-neox-20b":
        base_run.parameters["model"]["vocab_size"] = 50432
        base_run.parameters["tokenizer_name"] = "EleutherAI/gpt-neox-20b"
        base_run.parameters["tokenizer"]["kwargs"] = {
            "model_max_length": args.seq_len
        }


def launch_run(run, local_debug, seed):
    if local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(run), f=f)
    else:
        run = create_run(run)
        print(f"Launched seed {seed} with in {run.name}")


def build_model_arch(model_size):
    if model_size == "125M":
        model_cfg = {
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "lr": 0.0002,  # Setting lr low for now since ref is 10x chinchilla
            "weight_decay": 0.0002
        }
    elif model_size == "250M":
        model_cfg = {
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 16,
            "lr": 0.0002,
            "weight_decay": 0.0002
        }
    elif model_size == "1B":
        model_cfg = {
            "d_model": 2048,
            "n_heads": 16,
            "n_layers": 24,
            "lr": 0.0002,
            "weight_decay": 0.0002
        }
    elif model_size == "3B":
        model_cfg = {
            "d_model": 2560,
            "n_heads": 32,
            "n_layers": 32,
            "lr": 0.002,
            "weight_decay": 0.002
        }
    else:
        raise ValueError(f"Unknown model size {model_size}")
    return model_cfg


def build_dataset_base(dataset: str,
                       tokenizer_name: str,
                       seq_len: int,
                       num_tokens: int,
                       num_passes: str,
                       holdout: bool,
                       filter_suffix: Optional[str] = None):
    return f"s3://data-force-one-datasets/__unitystorage/catalogs/36798a58-e180-4029-8cd7-842e61841ef0/volumes/b9e4994e-997d-4cbf-b76b-e38ff5533785/{dataset}/{tokenizer_name}-seqlen-{seq_len}/52B-total-available-holdout-tokens-partition-sd-17/{'holdout' if holdout else 'train'}/{num_tokens}-tokens-from-{num_passes}-passes{f'-{filter_suffix}' if filter_suffix is not None else ''}/combined/mds"


def build_ref_base(num_tokens, num_params):
    return f"refp-{num_params}-reft-{num_tokens}"


def build_proxy_base(selection_algo,
                     proxy_num_tokens,
                     proxy_num_params,
                     full_batch_size,
                     num_pplx_filter,
                     ref_num_tokens=None,
                     ref_num_params=None):
    if num_pplx_filter > 0:
        assert num_pplx_filter < full_batch_size and num_pplx_filter > 512  # Hard setting 512 bs
    proxy_base = f"{selection_algo}{'-filpplx-' + str(num_pplx_filter) if num_pplx_filter > 0 else ''}-proxp-{proxy_num_params}-proxt-{proxy_num_tokens}-fb-{full_batch_size}"
    if selection_algo == "rho":
        assert (ref_num_params is not None and ref_num_tokens is not None and
                num_pplx_filter is not None)
        ref_run_base = build_ref_base(ref_num_tokens, ref_num_params)
        proxy_base += f"-{ref_run_base}"
    return proxy_base


def build_final_base(num_tokens, num_params):
    return f"finp-{num_params}-fint-{num_tokens}"


def assert_args(vals, args):
    for val in vals:
        assert val in vars(args) and vars(
            args)[val] is not None, f"Missing arg {val}"
