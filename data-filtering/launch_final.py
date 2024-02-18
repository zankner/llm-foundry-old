import argparse
import os

from mcli import RunConfig

from pretrain_utils import (CKPT_BASE, set_common_args, launch_run,
                            build_model_arch, build_dataset_base,
                            build_final_base, assert_args)

if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=16)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--priority",
                        type=str,
                        default="lowest",
                        choices=["lowest", "low", "medium"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        required=True)  # Add more later
    parser.add_argument("--overwrite-shuffle-seed", type=int, default=None)
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--global-batch-size",
                        type=int,
                        default=1024,
                        choices=[512, 1024])

    # Reference args
    parser.add_argument("--ref-lr", type=float, default=None)
    parser.add_argument("--ref-global-batch-size", type=int, default=512)
    parser.add_argument("--ref-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])
    parser.add_argument("--proxy-off-policy", action="store_true")

    # Final args
    parser.add_argument("--final-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M", "1B", "3B"])
    parser.add_argument("--final-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])

    # Selection args
    parser.add_argument("--selection-algo",
                        type=str,
                        choices=["offline", "rhol", "online", "baseline"
                                ])  # Treat baseline as a selection algo
    parser.add_argument("--selection-rank",
                        type=str,
                        default="hard",
                        choices=["hard", "mid",
                                 "easy"])  # Treat baseline as a selection algo
    parser.add_argument("--selection-rate",
                        type=float,
                        default=0.5,
                        choices=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--selection-scope", choices=["global", "local", "web"])

    # Data args
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt4-tiktoken",
                        choices=["gpt4-tiktoken", "gpt-neox-20b"])
    parser.add_argument("--available-holdout-tokens",
                        type=str,
                        default="52B",
                        choices=["26B", "52B"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-passes", type=str, required=True)
    parser.add_argument("--dataset",
                        type=str,
                        default="mpt",
                        choices=["mpt", "dolma", "pile", "pile-cc"])

    args = parser.parse_args()

    final_run_base = build_final_base(args.final_num_tokens,
                                      args.final_model_size)

    run_base = f"{args.dataset}-passes-{args.num_passes}-final-{args.final_model_size}-{args.final_num_tokens}"
    if args.selection_algo == "baseline":
        run_name = f"{run_base}-baseline"
    elif args.selection_algo == "offline":
        assert_args([
            "selection_rank", "selection_rate", "ref_model_size",
            "ref_num_tokens"
        ], args)
        run_name = f"{run_base}-offline-{args.selection_scope}-{args.selection_rank}-{args.selection_rate}-ref-{args.ref_model_size}-{args.ref_num_tokens}"
    elif args.selection_algo == "online":
        run_name = f"{run_base}-online-{args.selection_rank}-proxy-{args.proxy_model_size}-{args.proxy_num_tokens}"
    elif args.selection_algo == "rhols":
        run_name = f"{run_base}-rhols-{args.selection_rank}{'-zscore' if args.proxy_z_score else '-raw'}-{'-off-policy' if args.proxy_off_policy else ''}-proxy-{args.proxy_model_size}-{args.proxy_num_tokens}-ref-{args.ref_model_size}-{args.ref_num_tokens}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"data-filtering/yamls/pretrain_base.yaml")

        save_suffix = ""
        if args.selection_algo == "baseline":
            data_remote = build_dataset_base(
                args.dataset,
                args.tokenizer,
                args.seq_len,
                args.final_num_tokens,
                args.num_passes,
                args.available_holdout_tokens,
                holdout=False,
                seed=seed,
            )
        elif args.selection_algo == "offline":
            ref_lr = build_model_arch(
                args.ref_model_size
            )["lr"] if args.ref_lr is None else args.ref_lr
            base_run.parameters["ref_lr"] = ref_lr
            filter_suffix = f"offline-{args.selection_scope}-{args.selection_rank}-{args.selection_rate}-ref-{args.ref_model_size}-{args.ref_num_tokens}-bs-{args.global_batch_size}-lr-{ref_lr}-sd-{seed}"
            save_suffix = f"bs-{args.ref_global_batch_size}-lr-{ref_lr}"
            data_remote = build_dataset_base(
                args.dataset,
                args.tokenizer,
                args.seq_len,
                args.final_num_tokens,
                args.num_passes,
                args.available_holdout_tokens,
                holdout=False,
                filter_suffix=filter_suffix,
                seed=seed,
            )
        elif args.selection_algo == "online":
            remote_base = build_remote_base(
                num_holdout_tokens=args.holdout_num_tokens,
                dataset=args.dataset,
            )
            data_remote = os.path.join(
                remote_base, "pruned",
                f"{args.final_num_tokens}-final-tokens-pruned-from-{proxy_run_base}-sd-{seed}"
            )

            # Don't shuffle for proxy
            base_run.parameters["max_duration"] = "1ep"
            base_run.parameters["num_canonical_nodes"] = 1
            base_run.parameters["train_loader"]["dataset"]["shuffle"] = False
            del base_run.parameters["train_loader"]["dataset"][
                "shuffle_block_size"]
            del base_run.parameters["train_loader"]["dataset"]["shuffle_seed"]
            del base_run.parameters["train_loader"]["dataset"]["shuffle_algo"]

        save_base = os.path.join(CKPT_BASE, args.dataset,
                                 f"{args.tokenizer}-seqlen-{args.seq_len}",
                                 "final")
        print(save_suffix)
        set_common_args(args,
                        base_run,
                        run_name,
                        save_base,
                        data_remote,
                        args.final_model_size,
                        args.final_num_tokens,
                        seed,
                        save_suffix=save_suffix)

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            "final", f"final-params-{args.final_model_size}",
            f"final-tok-{args.final_num_tokens}",
            f"selection-algo-{args.selection_algo}",
            f"selection-rank-{args.selection_rank}",
            f"selection-rate-{args.selection_rate}",
            f"selection-scope-{args.selection_scope}",
            f"{'off-policy' if args.proxy_off_policy else 'on-policy'}",
            f"proxy-params-{args.proxy_model_size}",
            f"proxy-tok-{args.proxy_num_tokens}", f"ref-lr-{args.ref_lr}",
            f"ref-params-{args.ref_model_size}",
            f"ref-tok-{args.ref_num_tokens}", f"num-passes-{args.num_passes}",
            f"tokenizer-{args.tokenizer}", f"seq-len-{args.seq_len}"
        ]

        launch_run(base_run, args.local_debug, seed)
