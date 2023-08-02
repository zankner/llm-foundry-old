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
                        required=True)  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B"])
    parser.add_argument(
        "--full-batch-size",
        help="Batch size for points to be labeled that will then be pruned",
        type=int,
        choices=[512, 1024, 2048, 4096])
    parser.add_argument("--num-pplx-filter", type=int, default=0)
    parser.add_argument("--selection-algo",
                        type=str,
                        required=True,
                        choices=["rho", "hard-mine", "easy-mine"])

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B"])

    args = parser.parse_args()

    proxy_run_base = build_proxy_base(args.selection_algo,
                                      args.proxy_num_tokens,
                                      args.proxy_model_size,
                                      args.full_batch_size,
                                      args.num_pplx_filter, args.ref_num_tokens,
                                      args.ref_model_size)
    run_name = f"proxy-{args.dataset}-{proxy_run_base}-holdt-{args.holdout_num_tokens}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"amortized-obs/yamls/pretrain_base.yaml")

        remote_base = build_remote_base(
            num_holdout_tokens=args.holdout_num_tokens,
            dataset=args.dataset,
        )
        # handling different datasources for different selection algorithms
        data_remote = os.path.join(
            remote_base, "train",
            f"{build_ref_base(args.ref_num_tokens, args.ref_model_size)}-sd-{seed}"
            if args.selection_algo == "rho" else "base")

        # Handling overhead for proxy
        base_run.parameters["train_loader"]["num_workers"] = 1

        save_folder = os.path.join(CKPT_BASE, args.dataset, "proxy",
                                   f"{run_name}-sd-{seed}", "ckpts")

        assert args.full_batch_size % 512 == 0, "Full batch size must be a multiple of pruned batch size"
        token_multiplier = args.full_batch_size // 512
        set_common_args(args,
                        base_run,
                        run_name,
                        save_folder,
                        data_remote,
                        args.proxy_model_size,
                        args.proxy_num_tokens,
                        seed,
                        token_multiplier=token_multiplier)

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            "proxy", f"holdt-{args.holdout_num_tokens}",
            f"reft-{args.ref_model_size}", f"reft-{args.ref_num_tokens}",
            f"proxp-{args.proxy_model_size}", f"proxt-{args.proxy_num_tokens}",
            f"fb-{args.full_batch_size}", f"fillpplx-{args.num_pplx_filter}",
            args.selection_algo
        ]

        # Configuring the selection algorithm
        base_run.parameters["global_train_batch_size"] = args.full_batch_size

        selection_algorithm = {"num_subsample": 512}
        if args.selection_algo == "rho" and args.num_pplx_filter:
            selection_algorithm["num_pplx_filter"] = args.num_pplx_filter
        elif args.selection_algo == "hard-mine":
            selection_algorithm["ignore_ref"] = True
        elif args.selection_algo == "easy-mine":
            selection_algorithm["ignore_ref"] = True
            selection_algorithm["hard_examples"] = False

        base_run.parameters["algorithms"][
            "online_batch_selection"] = selection_algorithm

        launch_run(base_run, args.local_debug, seed)
