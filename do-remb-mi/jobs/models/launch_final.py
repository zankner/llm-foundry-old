import argparse
import tempfile
import os
import copy

from mcli import RunConfig, create_run
from composer.utils import get_file
import numpy as np
from omegaconf import OmegaConf

from utils import (build_domain_streams, build_final_name, build_proxy_name,
                   build_data_path)

replicate_proportions = [
    0.6057, 0.0046, 0.0224, 0.1019, 0.0036, 0.0113, 0.0072, 0.0047, 0.0699,
    0.0018, 0.0093, 0.0061, 0.0062, 0.0134, 0.0502, 0.0274, 0.0063, 0.0070
]  # Weights from DoReMI paper for 280M ref model

WEIGHTS_BASE = "oci://mosaicml-internal-doremi/pile/proxy-weights"


def load_weights(weight_file: str):
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(weight_file, tmp_file.name, overwrite=True)
        weights = np.load(tmp_file.name)
    return weights


def get_proportions(args, seed):
    if args.domain_source == "baseline":
        return [None] * args.num_domains
    if args.domain_source == "replicate":
        return replicate_proportions
    else:
        remote_base = os.path.join("oci://mosaicml-internal-doremi",
                                   args.dataset, "proxy-weights")
        proxy_args = copy.deepcopy(args)
        proxy_args.subsample_dist = args.ref_subsample_dist
        proxy_args.model_size = args.proxy_model_size
        _, domain_types = build_data_path(proxy_args, mode="token-ref-loss")
        proxy_name = build_proxy_name(proxy_args,
                                      domain_types) + f"-seed-{seed}"
        weights_path = os.path.join(remote_base, proxy_name,
                                    f"ba-{args.proxy_step}",
                                    "average_domain_weights.npy")
        return [float(weight) for weight in list(load_weights(weights_path))]


if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r8z6")
    parser.add_argument("--ngpus", type=int, default=16)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--model-size", type=str, default="1B")

    # DoReMi args
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1e-4)
    parser.add_argument("--init-dist", type=str, default="uniform")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--proxy-model-size", type=str, default="125M")

    # Final args
    parser.add_argument("--domain-source",
                        type=str,
                        choices=["baseline", "replicate", "doremi"])
    parser.add_argument("--proxy-step", type=int, default=100000)

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt-neox-20b-seqlen-2048")
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--subsample-dist",
                        type=str,
                        required=True,
                        choices=["uniform"])
    parser.add_argument("--ref-subsample-dist", type=str, default="baseline")
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["100K", "all"])
    parser.add_argument("--not-embed", action="store_true")
    args = parser.parse_args()

    if args.not_embed:
        data_remote_dir = "data-sources"
    else:
        data_remote_dir = f"{args.num_domains}-clusters"

    remote_base, domain_types = build_data_path(args, mode="pre-concat")

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/final/pretrain_final.yaml")

        proportions = get_proportions(args, seed)
        domain_streams = build_domain_streams(args.num_domains,
                                              remote_base,
                                              proportions=proportions)

        base_run.name = f"zack-final-{args.model_size}-step-{args.num_samples}-sd-{seed}".lower(
        )
        base_run.parameters["run_name"] = build_final_name(args, domain_types)

        base_run.cluster = args.cluster
        base_run.gpu_num = args.ngpus

        if args.preemptible:
            assert args.autoresume is True, "Preemptible training requires autoresume"
            base_run.scheduling = {"resumable": True, "priority": "low"}
            base_run.parameters["autoresume"] = True
        else:
            base_run.parameters["autoresume"] = args.autoresume

        base_run.parameters["global_seed"] = seed

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            args.model_size,
            f"step-size-{args.step_size}",
            f"smoothing-{args.smoothing}",
            f"dataset-{args.dataset}",
            f"domain-{domain_types}",
            f"ref-subsample-dist-{args.ref_subsample_dist}",
            f"subsample-dist-{args.subsample_dist}",
            f"samples-{args.num_samples}",
            f"warmup-steps-{args.warmup_steps}",
            f"domain-source-{args.domain_source}",
        ]

        base_run.parameters[
            "device_train_microbatch_size"] = args.device_batch_size
        base_run.parameters["device_eval_batch_size"] = args.device_batch_size

        base_run.parameters["train_loader"]["dataset"][
            "streams"] = domain_streams

        if args.local_debug:
            with open("debug-final.yaml", "w") as f:
                OmegaConf.save(config=OmegaConf.create(base_run.parameters),
                               f=f)
        else:
            run = create_run(base_run)
            print(f"Launched final training for seed {seed} with in {run.name}")