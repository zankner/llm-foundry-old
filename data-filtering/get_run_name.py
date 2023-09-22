import argparse
from pretrain_utils import (build_ref_base, build_proxy_base, build_final_base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Proxy args
    parser.add_argument("--proxy-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--proxy-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])
    parser.add_argument(
        "--full-batch-size",
        help="Batch size for points to be labeled that will then be pruned",
        type=int,
        choices=[1024, 2048, 4096])
    parser.add_argument("--num-pplx-filter", type=int, default=0)
    parser.add_argument("--selection-algo",
                        type=str,
                        choices=["rho", "hard-mine", "easy-mine", "baseline"
                                ])  # Treat baseline as a selection algo

    # Final args
    parser.add_argument("--final-model-size",
                        type=str,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--final-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Data args
    parser.add_argument("--dataset", type=str, default="pile", choices=["pile"])
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Misc
    parser.add_argument("--run-type",
                        type=str,
                        required=True,
                        choices=["reference", "proxy", "final"])

    # Seed args
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    # Building the run name
    if args.run_type == "final":
        final_base = build_final_base(args.final_num_tokens,
                                      args.final_model_size)
        if args.selection_algo == "baseline":
            run_name = f"final-{args.dataset}-baseline-{final_base}"
        else:
            proxy_run_base = build_proxy_base(
                args.selection_algo, args.proxy_num_tokens,
                args.proxy_model_size, args.full_batch_size,
                args.num_pplx_filter, args.ref_num_tokens, args.ref_model_size)
            fmt_proxy_run_base = proxy_run_base.replace(
                f"{args.selection_algo}-", "")
            run_name = f"final-{args.dataset}-{args.selection_algo}-{final_base}-{fmt_proxy_run_base}"
    elif args.run_type == "proxy":
        model_size = args.proxy_model_size
        num_tokens = args.proxy_num_tokens
        proxy_run_base = build_proxy_base(
            args.selection_algo, args.proxy_num_tokens, args.proxy_model_size,
            args.full_batch_size, args.num_pplx_filter, args.ref_num_tokens,
            args.ref_model_size)
        run_name = f"proxy-{args.dataset}-{proxy_run_base}"
    elif args.run_type == "reference":
        ref_base = build_ref_base(args.ref_num_tokens, args.ref_model_size)
        run_name = f"ref-{args.dataset}-{ref_base}"

    run_name = f"{run_name}-holdt-{args.holdout_num_tokens}-sd-{args.seed}"
    print("Train run name:")
    print(run_name)
    print()
    print("Eval final name:")
    print(f"eval-final-{run_name}")
    print()
    print("Eval sweep name:")
    print(f"eval-sweep-{run_name}")