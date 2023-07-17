import argparse

from mcli import RunConfig

from utils import (build_domain_streams, build_final_name, build_proxy_name,
                   get_remote_data_path, get_proxy_weights, launch_run,
                   set_common_args, PILE_BASELINE_PROPORTIONS)

# These are the weights reported in the paper
replicate_125M_proportions = [
    0.20163412392139435, 0.10023175925016403, 0.13405060768127441,
    0.1319042444229126, 0.07527627050876617, 0.09094192832708359,
    0.05040840804576874, 0.04523293673992157, 0.024078955873847008,
    0.03058076836168766, 0.019306106492877007, 0.0025820545852184296,
    0.0365639366209507, 0.002238696441054344, 0.00817348062992096,
    0.0037490464746952057, 0.016184765845537186, 0.004014396108686924,
    0.008683200925588608, 0.007775607518851757, 0.004500322509557009,
    0.0018897337140515447
]
replicate_250M_proportions = [
    0.6057, 0.0046, 0.0224, 0.1019, 0.0036, 0.0113, 0.0072, 0.0047, 0.0699,
    0.0018, 0.0093, 0.0061, 0.0062, 0.0134, 0.0502, 0.0274, 0.0063, 0.0070
]

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
    parser.add_argument("--model-size", type=str, default="1B", choices=["1B"])
    parser.add_argument("--device-batch-size", type=int, default=32)

    # DoReMi args
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1e-4)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--proxy-model-size",
                        type=str,
                        default="125M",
                        choices=["125M", "250M"])
    parser.add_argument("--proxy-num-samples",
                        type=str,
                        choices=["10K", "25K", "100K", "all"])

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt-neox-20b-seqlen-2048")
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--domain-weight-source",
                        type=str,
                        required=True,
                        choices=[
                            "baseline", "125M-replicate", "250M-replicate",
                            "pile", "proxy"
                        ])
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["25K", "100K", "all"])
    parser.add_argument("--not-embed", action="store_true")

    args = parser.parse_args()

    run_name = build_final_name(args)

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/final/pretrain_final.yaml")

        remote_base = get_remote_data_path(args, "final", seed)
        if args.domain_weight_source == "baseline":
            proportions = [None] * args.num_domains
        elif args.domain_weight_source == "125M-replicate":
            proportions = replicate_125M_proportions
        elif args.domain_weight_source == "250M-replicate":
            proportions = replicate_250M_proportions
        elif args.domain_weight_source == "pile":
            proportions = PILE_BASELINE_PROPORTIONS
        elif args.domain_weight_source == "proxy":
            proxy_run_name = build_proxy_name(
                args, args.iter, args.proxy_model_size,
                args.proxy_num_samples) + f"-sd-{seed}"
            proportions = get_proxy_weights(proxy_run_name, args.dataset)
        domain_streams = build_domain_streams(args.num_domains,
                                              remote_base,
                                              proportions=proportions)

        base_run = set_common_args(args, base_run, run_name, seed,
                                   args.proxy_model_size,
                                   args.proxy_num_samples, domain_streams)

        launch_run(base_run, local_debug=args.local_debug, seed=seed)