import argparse
import os

from mcli import RunConfig

from pretrain_utils import (CKPT_BASE, set_common_args, launch_run,
                            build_remote_base, build_ref_base, build_proxy_base,
                            build_final_base)

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
    parser.add_argument("--selection-algo",
                        type=str,
                        choices=["rho", "hard-mine", "easy-mine"])

    # Final args
    parser.add_argument("--final-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--final-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])
    parser.add_argument("--is-baseline", action="store_true")

    # Data args
    parser.add_argument("--dataset", type=str, default="pile", choices=["pile"])
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B"])

    args = parser.parse_args()

    if args.is_baseline:
        suffix = "baseline"
    else:
        assert args.holdout_num_tokens is not None
        proxy_run_base = build_proxy_base(args.selection_algo,
                                          args.proxy_num_tokens,
                                          args.proxy_model_size,
                                          args.full_batch_size,
                                          args.num_pplx_filter)
        if args.selection_algo == "rho":
            ref_run_base = build_ref_base(args.ref_num_tokens,
                                          args.ref_model_size)
            proxy_run_base += f"-{ref_run_base}"
        suffix = proxy_run_base
    final_run_base = build_final_base(args.final_num_tokens,
                                      args.final_model_size)
    run_name = f"final-{args.dataset}-{final_run_base}-{suffix}-holdt-{args.holdout_num_tokens}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"amortized-obs/yamls/pretrain_base.yaml")

        if args.is_baseline:
            # Making the choice to select baseline from the partitioned data for consistency
            remote_base = build_remote_base(
                num_holdout_tokens=args.ref_num_tokens,
                dataset=args.dataset,
            )
            data_remote = os.path.join(
                remote_base,
                "train",
                "base",
            )
        else:
            remote_base = build_remote_base(
                num_holdout_tokens=args.ref_num_tokens,
                dataset=args.dataset,
            )
            data_remote = os.path.join(
                remote_base, "pruned",
                f"{args.final_num_tokens}-final-tokens-pruned-from-{proxy_run_base}"
            )

            # Don't shuffle for proxy
            base_run.parameters["max_duration"] = "1ep"
            base_run.parameters["num_canonical_nodes"] = 1
            base_run.parameters["train_loader"]["dataset"]["shuffle"] = False
            del base_run.parameters["train_loader"]["dataset"][
                "shuffle_block_size"]
            del base_run.parameters["train_loader"]["dataset"]["shuffle_seed"]
            del base_run.parameters["train_loader"]["dataset"]["shuffle_algo"]

        save_folder = os.path.join(CKPT_BASE, "final", f"{run_name}-sd-{seed}",
                                   "ckpts")

        set_common_args(args, base_run, run_name, save_folder, data_remote,
                        args.final_model_size, args.final_num_tokens, seed)

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            "final", f"fp-{args.final_model_size}",
            f"ft-{args.final_num_tokens}"
        ]
        if args.is_baseline:
            base_run.parameters["loggers"]["wandb"]["tags"] += ["baseline"]
        else:
            base_run.parameters["loggers"]["wandb"]["tags"] += [
                f"holdt-{args.holdout_num_tokens}",
                f"refp-{args.ref_model_size}", f"reft-{args.ref_num_tokens}",
                f"proxp-{args.proxy_model_size}",
                f"proxt-{args.proxy_num_tokens}", f"fb-{args.full_batch_size}",
                f"fillpplx-{args.num_pplx_filter}", args.selection_algo
            ]

        launch_run(base_run, args.local_debug, seed)
