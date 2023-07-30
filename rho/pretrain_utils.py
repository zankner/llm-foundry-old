import os

from omegaconf import OmegaConf as om
from mcli import create_run

CKPT_BASE = "oci://mosaicml-internal-checkpoints/zack/rho/"


def set_common_args(args,
                    base_run,
                    run_name,
                    save_folder,
                    data_remote,
                    model_size,
                    duration,
                    seed,
                    token_multiplier=1):
    # Set run name
    base_run.name = run_name.lower()[:56]  # Mcli things
    base_run.parameters["run_name"] = run_name

    # Set seed
    base_run.parameters["global_seed"] = seed

    # Set compute
    base_run.cluster = args.cluster
    base_run.gpu_num = args.ngpus
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

    # Set modeling args
    model_cfg = build_model_arch(model_size)
    base_run.parameters["model"]["d_model"] = model_cfg["d_model"]
    base_run.parameters["model"]["n_heads"] = model_cfg["n_heads"]
    base_run.parameters["model"]["n_layers"] = model_cfg["n_layers"]

    # Set ckpt save folder
    base_run.parameters["save_folder"] = save_folder

    # Set data args
    base_run.parameters["train_loader"]["dataset"]["remote"] = data_remote

    # Common wandb tags
    base_run.parameters["loggers"]["wandb"]["tags"] += [
        f"dataset-{args.dataset}", f"seed-{seed}"
    ]

    # Handle preemption
    if args.preemptible:
        assert args.autoresume is True, "Preemptible training requires autoresume"
        base_run.scheduling = {"resumable": True, "priority": "low"}
        base_run.parameters["autoresume"] = True
    else:
        base_run.parameters["autoresume"] = args.autoresume

    # Set batch size
    base_run.parameters["device_train_microbatch_size"] = args.device_batch_size
    base_run.parameters["device_eval_batch_size"] = args.device_batch_size

    # Set training duration
    if duration == "2B":
        duration_tokens = 2_000_000_000
    elif duration == "5B":
        duration_tokens = 5_000_000_000
    elif duration == "20B":
        duration_tokens = 20_000_000_000
    elif duration == "26B":
        duration_tokens = 26_000_000_000
    duration_tokens *= token_multiplier
    base_run.parameters["max_duration"] = f"{duration_tokens}tok"

    base_run.parameters["eval_interval"] = "1000ba"
    base_run.parameters["save_interval"] = "500ba"


def launch_run(run, local_debug, seed):
    if local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(run.parameters), f=f)
    else:
        run = create_run(run)
        print(f"Launched seed {seed} with in {run.name}")


def build_model_arch(model_size):
    if model_size == "125M":
        model_cfg = {"d_model": 768, "n_heads": 12, "n_layers": 12}
    elif model_size == "250M":
        model_cfg = {"d_model": 1024, "n_heads": 16, "n_layers": 16}
    elif model_size == "1B":
        model_cfg = {"d_model": 2048, "n_heads": 16, "n_layers": 24}
    else:
        raise ValueError(f"Unknown model size {model_size}")
    return model_cfg


def build_remote_base(num_holdout_tokens, dataset, seed):
    return os.path.join("oci://mosaicml-internal-amortized-obs", "rho", dataset,
                        f"{num_holdout_tokens}-holdout-tokens-sd-{seed}")


def build_ref_base(num_tokens, num_params):
    return f"{num_params}-hop-{num_tokens}-hot"


def build_proxy_base(sel_alg, num_tokens, num_params, full_batch_size):
    return f"{sel_alg}-{num_params}-pp-{num_tokens}-pt-{full_batch_size}-fb"


def build_final_base(num_tokens, num_params):
    return f"{num_params}-fp-{num_tokens}-ft"

# domain streams handle different remotes for different app
def build_domain_streams(num_domains: int,
                         remote_base: str,
                         proportions: Optional[List[float]] = None):

    train_streams = {
        f"domain-{domain_id}": {
            "local": f"/tmp/dataset/domain-{domain_id}/",
            "remote": os.path.join(remote_base, f"domain-{domain_id}"),
            "split": "train",
        } for domain_id in range(num_domains)
    }
    if proportions is not None:
        train_streams = {
            domain_id: {
                **stream, "proportion": proportion
            } for (domain_id,
                  stream), proportion in zip(train_streams.items(), proportions)
        }
    return train_streams
