import argparse
import os

from mcli import RunConfig

from pretrain_utils import (
    CKPT_BASE,
    set_common_args,
    launch_run,
    build_remote_base,
    build_seed_name,
    build_domain_streams,
	duration_to_tokens,
)

def run_seed(args, sd):
    run_name = build_seed_name(args.dataset,
                               args.domain_type,
                               args.model_size,
                               args.num_tokens,
                               args.warmup_duration)

    base_run = RunConfig.from_file(f"brick/yamls/pretrain_base.yaml")

    remote_base = "oci://mosaicml-internal-doremi/pile/pre-concat/gpt-neox-20b-seqlen-2048/22-clusters/base/all-samples-baseline-sd-17/"
    save_folder = os.path.join(CKPT_BASE, "reference", f"{run_name}-sd-{sd}", "ckpts")

    domain_streams = build_domain_streams(args.num_domains,
                                          remote_base,
                                          proportions=None)

    set_common_args(args,
                    sd,
                    base_run,
                    run_name,
                    save_folder,
                    domain_streams,
                    args.domain_type,
                    args.num_domains,
                    args.model_size,
                    args.num_tokens,
                    args.warmup_duration)
    total_tokens = duration_to_tokens(args.num_tokens)
    base_run.parameters["callbacks"]["stop_time"] = f"{int(args.warmup_duration * total_tokens)}tok"

    return launch_run(base_run, args.local_debug, sd)

if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--device-batch-size", type=int, default=32)

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--domain-type",
                        type=str,
                        default="clusters",
                        choices=["clusters", "provenance"])
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])
    parser.add_argument("--warmup-duration",
                        type=float,
                        required=True)

    args = parser.parse_args()

    for sd in args.seeds:
        run_seed(args, sd)
