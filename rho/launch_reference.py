import argparse
import os

from mcli import RunConfig

from pretrain_utils import (CKPT_BASE, set_common_args, launch_run,
                            build_remote_base, build_ref_base)

if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=16)
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
    parser.add_argument("--num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])

    args = parser.parse_args()

    run_base = build_ref_base(args.num_tokens, args.model_size)
    run_name = f"ref-{args.dataset}-{run_base}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(f"rho/yamls/pretrain_base.yaml")

        data_remote = os.path.join(
            build_remote_base(num_holdout_tokens=args.num_tokens,
                              dataset=args.dataset,
                              seed=seed), "holdout")

        save_folder = os.path.join(CKPT_BASE, "reference",
                                   f"{run_name}-sd-{seed}", "ckpts")

        set_common_args(args, base_run, run_name, save_folder, data_remote,
                        args.model_size, args.num_tokens, seed)

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            "ref", f"hop-{args.model_size}", f"hot-{args.num_tokens}"
        ]

        launch_run(base_run, args.local_debug, seed)
