import argparse
import os

from mcli import RunConfig

from pretrain_utils import (CKPT_BASE, set_common_args, launch_run,
                            build_remote_base, build_ref_base, build_proxy_base)

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

    # Reference args
    parser.add_argument("--ref-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])
    parser.add_argument(
        "--full-batch-size",
        help="Batch size for points to be labeled that will then be pruned",
        type=int,
        choices=[1024, 2048, 4096])

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--device-batch-size", type=int, default=32)

    args = parser.parse_args()

    ref_run_base = build_ref_base(args.ref_num_tokens, args.ref_model_size)
    proxy_run_base = build_proxy_base(args.proxy_num_tokens,
                                      args.proxy_model_size,
                                      args.full_batch_size)
    run_name = f"proxy-{args.dataset}-{proxy_run_base}-{ref_run_base}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(f"rho/yamls/pretrain_base.yaml")

        remote_base = build_remote_base(num_holdout_tokens=args.ref_num_tokens,
                                        dataset=args.dataset,
                                        seed=seed)
        data_remote = os.path.join(remote_base, "train", ref_run_base)

        save_folder = os.path.join(CKPT_BASE, "proxy", f"{run_name}-sd-{seed}",
                                   "ckpts")

        set_common_args(args, base_run, run_name, save_folder, data_remote,
                        args.proxy_model_size, args.proxy_num_tokens, seed)

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            f"hop-{args.ref_model_size}", f"hot-{args.ref_num_tokens}",
            f"pp-{args.proxy_model_size}", f"pt-{args.proxy_num_tokens}",
            f"fb-{args.full_batch_size}"
        ]

        # Configuring the RHO algorithm
        base_run.parameters["global_train_batch_size"] = args.full_batch_size
        base_run.parameters["algorithms"]["rho"] = {"num_subsample": 512}

        launch_run(base_run, args.local_debug, seed)
