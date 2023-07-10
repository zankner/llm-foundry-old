import argparse

from mcli import RunConfig

from utils import (build_domain_streams, get_remote_data_path, build_proxy_name,
                   launch_run, set_common_args)

# CHANGE BACK THE SAMPLING DIR TO UNIFORM INSTEAD OF BASELINE
# change back the remote data base to be iter and not iter - 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System args
    parser.add_argument("--cluster", type=str, default="r8z6")
    parser.add_argument("--ngpus", type=int, default=16)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--model-size",
                        type=str,
                        default="125M",
                        choices=["125M", "250M"])
    parser.add_argument("--device-batch-size", type=int, default=32)

    # DoReMi args
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--iter", type=int, default=1)

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt-neox-20b-seqlen-2048")
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["1K", "5K", "10K", "25K", "100K", "all"])
    parser.add_argument("--not-embed", action="store_true")

    args = parser.parse_args()

    run_name = f"replicate-{build_proxy_name(args, args.iter, args.model_size,args.num_samples)}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/proxy/pretrain_proxy.yaml")

        remote_base = get_remote_data_path(args, "proxy", seed)
        proportions = [1 / args.num_domains] * args.num_domains
        domain_streams = build_domain_streams(args.num_domains, remote_base,
                                              proportions)

        # DomainWeightSetter params
        base_run.parameters["algorithms"]["doremi"][
            "num_domains"] = args.num_domains
        base_run.parameters["algorithms"]["doremi"][
            "step_size"] = args.step_size
        base_run.parameters["algorithms"]["doremi"][
            "smoothing"] = args.smoothing
        base_run.parameters["algorithms"]["doremi"][
            "warmup_steps"] = args.warmup_steps
        base_run.parameters["algorithms"]["doremi"]["doremi_iter"] = args.iter

        base_run = set_common_args(args, base_run, run_name, seed,
                                   args.model_size, args.num_samples,
                                   domain_streams)

        launch_run(base_run, local_debug=args.local_debug, seed=seed)
