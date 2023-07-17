import os
from typing import List, Optional
import tempfile

import numpy as np
from omegaconf import OmegaConf
from composer.utils import get_file
from mcli import create_run

PILE_BASELINE_PROPORTIONS = [
    0.09453842840643951, 0.14917909233033527, 0.10094214543052439,
    0.10655481967324043, 0.12080817542576074, 0.057931398201810855,
    0.04899461799656484, 0.06609301371629318, 0.033852550742558066,
    0.028786516694716036, 0.030963661397225353, 0.014752223203858505,
    0.06490474172635509, 0.02527727404201975, 0.014990856135336905,
    0.006260514186577765, 0.0101892107589665, 0.007291182280087811,
    0.00928314413519068, 0.004190338888072547, 0.002578251119249607,
    0.0016378435088161597
]


def set_common_args(args, base_run, run_name, seed, proxy_ref_size,
                    proxy_ref_samples, domain_streams):
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
        base_run.gpu_type = "a100_80gb"
    else:
        base_run.gpu_type = "a100_40gb"

    # Set modeling args
    model_cfg = build_model_cfg(args.model_size)
    base_run.parameters["model"]["d_model"] = model_cfg["d_model"]
    base_run.parameters["model"]["n_heads"] = model_cfg["n_heads"]
    base_run.parameters["model"]["n_layers"] = model_cfg["n_layers"]

    # Set data args
    base_run.parameters["train_loader"]["dataset"]["streams"] = domain_streams

    # Set batch size
    base_run.parameters["device_train_microbatch_size"] = args.device_batch_size
    base_run.parameters["device_eval_batch_size"] = args.device_batch_size

    # Set eval/ckpt freq
    if args.num_samples in ["10K", "25K"]:
        base_run.parameters["eval_interval"] = "1000ba"
        base_run.parameters["save_interval"] = "500ba"
    elif args.num_samples == "100K":
        base_run.parameters["eval_interval"] = "2000ba"
        base_run.parameters["save_interval"] = "1000ba"
    else:
        raise ValueError(f"Invalid num_samples {args.num_samples}")

    # Common wandb tags
    base_run.parameters["loggers"]["wandb"]["tags"] += [
        f"dataset-{args.dataset}",
        f"data-source-{get_data_source(args)}",
        f"nprs-{proxy_ref_samples}",
        f"prp-{proxy_ref_size}",
        f"iter-{args.iter}",
    ]

    # Handle preemption
    if args.preemptible:
        assert args.autoresume is True, "Preemptible training requires autoresume"
        base_run.scheduling = {"resumable": True, "priority": "low"}
        base_run.parameters["autoresume"] = True
    else:
        base_run.parameters["autoresume"] = args.autoresume

    return base_run


def launch_run(run, local_debug, seed):
    if local_debug:
        with open("debug.yaml", "w") as f:
            OmegaConf.save(config=OmegaConf.create(run.parameters), f=f)
    else:
        run = create_run(run)
        print(f"Launched seed {seed} with in {run.name}")


def get_remote_data_path(args, run_type, seed):
    if run_type == "proxy":
        if args.iter == 1:
            data_name = os.path.join(
                "token-ref-loss",
                f"{args.num_samples}-samples-prp-{args.model_size}-baseline")
        else:
            proxy_desc = proxy_descriptor(
                args.iter - 1,  # Safe to use iter since proxy model
                args.step_size,
                args.smoothing,
                args.warmup_steps,
            )
            data_name = os.path.join(
                "token-ref-loss",
                f"{args.num_samples}-samples-prp-{args.model_size}-proxy-{proxy_desc}"
            )
    else:
        data_name = os.path.join("base", f"{args.num_samples}-samples-baseline")
    data_source = get_data_source(args)
    return os.path.join("oci://mosaicml-internal-doremi", args.dataset,
                        "pre-concat", args.tokenizer, data_source,
                        f"{data_name}-sd-{seed}")


def get_data_source(args):
    return "data-sources" if args.not_embed else f"{args.num_domains}-clusters"


def proxy_ref_preamble(num_samples, num_params):
    return f"nprs-{num_samples}-ppr-{num_params}"


def proxy_descriptor(iteration, step_size, smoothing, warmup_steps):
    return f"iter-{iteration}-ss-{step_size}-sm-{smoothing}-ws-{warmup_steps}"


def build_ref_name(args):
    data_source = get_data_source(args)
    prefix = proxy_ref_preamble(args.num_samples, args.model_size)
    if args.iter == 1:
        domain_weights = "baseline"
    else:
        domain_weights = proxy_descriptor(args.iter, args.step_size,
                                          args.smoothing, args.warmup_steps)
    return f"ref-{data_source}-{prefix}-domain-weights-{domain_weights}", domain_weights


def build_proxy_name(args, iteration, proxy_model_size, proxy_num_samples):
    data_source = get_data_source(args)
    prefix = proxy_ref_preamble(proxy_num_samples, proxy_model_size)
    proxy_desc = proxy_descriptor(iteration, args.step_size, args.smoothing,
                                  args.warmup_steps)
    return f"proxy-{data_source}-{prefix}-{proxy_desc}"


def build_final_name(args):
    data_source = get_data_source(args)
    if args.domain_weight_source == "proxy":
        proxy_prefix = proxy_ref_preamble(args.proxy_num_samples,
                                          args.proxy_model_size)
        proxy_desc = proxy_descriptor(args.iter, args.step_size, args.smoothing,
                                      args.warmup_steps)
        domain_weights = f"{proxy_prefix}-{proxy_desc}"
    else:
        domain_weights = args.domain_weight_source
    final_prefix = f"nfs-{args.num_samples}-nfp-{args.model_size}"
    return f"final-{data_source}-{final_prefix}-domain-weights-{domain_weights}"


def build_model_cfg(model_size):
    if model_size == "125M":
        model_cfg = {"d_model": 768, "n_heads": 12, "n_layers": 12}
    elif model_size == "250M":
        model_cfg = {"d_model": 1024, "n_heads": 16, "n_layers": 16}
    elif model_size == "1B":
        model_cfg = {"d_model": 2048, "n_heads": 16, "n_layers": 24}
    else:
        raise ValueError(f"Unknown model size {model_size}")
    return model_cfg


def get_proxy_weights(proxy_run_name, dataset):
    # CHANGE BACK TO HAVING A DATASET PREFIX
    # ie: f"oci://mosaicml-internal-checkpoints/zack/DoReMi/{dataset}/proxy/{proxy_run_name}/average-domain-weights/final/average.npy"
    weight_file = f"oci://mosaicml-internal-checkpoints/zack/DoReMi/proxy/{proxy_run_name}/domain-weights/final/average_domain_weights.npy"
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(weight_file, tmp_file.name, overwrite=True)
        weights = np.load(tmp_file.name)
    return [float(weight) for weight in weights]


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
