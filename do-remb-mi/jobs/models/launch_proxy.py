import argparse
import copy

from mcli import RunConfig, create_run
from omegaconf import OmegaConf

from utils import build_domain_streams, build_data_path, build_proxy_name

# CHANGE BACK THE SAMPLING DIR TO UNIFORM INSTEAD OF BASELINE
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
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--model-size", type=str, default="125M")

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
    parser.add_argument("--subsample-dist",
                        type=str,
                        required=True,
                        choices=["uniform"])
    parser.add_argument("--ref-subsample-dist", type=str, default="baseline")
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["100K", "all"])
    parser.add_argument("--not-embed", action="store_true")
    args = parser.parse_args()

    ref_args = copy.deepcopy(args)
    ref_args.subsample_dist = args.ref_subsample_dist
    remote_base, domain_types = build_data_path(ref_args, mode="token-ref-loss")
    if args.subsample_dist == "uniform":
        proportions = [1 / args.num_domains] * args.num_domains
    domain_streams = build_domain_streams(args.num_domains,
                                          remote_base,
                                          proportions=proportions)

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/proxy/pretrain_proxy.yaml")

        base_run.name = f"zack-proxy-{args.model_size}-{args.subsample_dist}-{args.num_samples}-sd-{seed}".lower(
        )
        base_run.parameters["run_name"] = build_proxy_name(args, domain_types)

        base_run.cluster = args.cluster
        base_run.gpu_num = args.ngpus

        if args.preemptible:
            # assert args.autoresume is True, "Preemptible training requires autoresume"
            base_run.scheduling = {"resumable": True, "priority": "low"}
            base_run.parameters["autoresume"] = True
        else:
            base_run.parameters["autoresume"] = args.autoresume

        base_run.parameters["global_seed"] = seed

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            args.model_size,
            f"step-size-{args.step_size}",
            f"smoothing-{args.smoothing}",
            f"dataset-{args.dataset}",
            f"domain-{domain_types}",
            f"ref-subsample-dist-{args.ref_subsample_dist}",
            f"subsample-dist-{args.subsample_dist}",
            f"samples-{args.num_samples}",
            f"warmup-steps-{args.warmup_steps}",
        ]

        # No microbatching allowed for proxy run
        base_run.parameters[
            "device_train_microbatch_size"] = args.device_batch_size
        base_run.parameters["device_eval_batch_size"] = args.device_batch_size

        base_run.parameters["train_loader"]["dataset"][
            "streams"] = domain_streams

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

        if args.local_debug:
            with open("debug-proxy.yaml", "w") as f:
                OmegaConf.save(config=OmegaConf.create(base_run.parameters),
                               f=f)
        else:
            run = create_run(base_run)
            print(f"Launched proxy training for seed {seed} with in {run.name}")