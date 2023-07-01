import argparse

from mcli import RunConfig

from utils import (set_common_args, get_remote_data_path, launch_run,
                   build_ref_name, build_proxy_name, build_domain_streams,
                   get_proxy_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System args
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--no-log-domain-loss", action="store_true")
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--model-size",
                        type=str,
                        default="125M",
                        choices=["125M", "250M"])
    parser.add_argument("--device-batch-size", type=int, default=32)

    # Reference args
    parser.add_argument("--iter", type=int, default=1)

    # Proxy args
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt-neox-20b-seqlen-2048")
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--domain-weight-source",
                        type=str,
                        required=True,
                        choices=["baseline", "proxy"])
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["1K", "5K", "10K", "25K", "100K", "all"])
    parser.add_argument("--not-embed", action="store_true")

    args = parser.parse_args()

    assert (args.iter > 1) ^ (
        args.domain_weight_source == "baseline"
    ), "Either iterated DoReMi or baseline domain distribution must be used"

    if args.domain_weight_source == "proxy":
        assert args.iter > 1, "Must have more than one iteration to use iterated DoReMi"
        args.proxy_iter = args.iter - 1

    run_name, domain_weight_source = build_ref_name(args)

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/reference/pretrain_ref.yaml")

        remote_base = get_remote_data_path(args, "ref", seed)
        if args.domain_weight_source == "baseline":
            proportions = [None] * args.num_domains
        elif args.domain_weight_source == "proxy":
            proxy_run_name = build_proxy_name(args, args.proxy_iter,
                                              args.model_size,
                                              args.num_samples) + f"-sd-{seed}"
            proprotions = get_proxy_weights(proxy_run_name, args.dataset)
        domain_streams = build_domain_streams(args.num_domains,
                                              remote_base,
                                              proportions=proportions)

        base_run = set_common_args(args, base_run, run_name, seed,
                                   args.model_size, args.num_samples,
                                   domain_streams)

        # Handle logging of domain wise losses
        if not args.no_log_domain_loss:
            base_run.parameters["callbacks"]["log_domain_loss"] = {
                "num_domains": args.num_domains
            }

        launch_run(base_run, local_debug=args.local_debug, seed=seed)
