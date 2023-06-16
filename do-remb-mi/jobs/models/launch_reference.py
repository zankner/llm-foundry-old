import argparse

from omegaconf import OmegaConf
from mcli import RunConfig, create_run

from utils import build_domain_streams, build_data_path, build_ref_name
# revert -n 1 and eval_first=False
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
    parser.add_argument("--model-size", type=str, default="125M")
    parser.add_argument("--device-batch-size", type=int, default=32)

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt-neox-20b-seqlen-2048")
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--subsample-dist",
                        type=str,
                        required=True,
                        choices=["baseline", "uniform"])
    parser.add_argument("--num-samples",
                        type=str,
                        required=True,
                        choices=["100K", "all"])
    parser.add_argument("--not-embed", action="store_true")

    args = parser.parse_args()

    remote_base, domain_types = build_data_path(args, mode="pre-concat")
    if args.subsample_dist == "uniform":
        proportions = [1 / args.num_domains] * args.num_domains
    else:
        proportions = None
    domain_streams = build_domain_streams(args.num_domains,
                                          remote_base,
                                          proportions=proportions)

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/reference/pretrain_ref.yaml")

        base_run.name = f"zack-ref-{args.model_size}-{args.subsample_dist}-{args.num_samples}-sd-{seed}".lower(
        )
        base_run.parameters["run_name"] = build_ref_name(args, domain_types)

        base_run.cluster = args.cluster
        base_run.gpu_num = args.ngpus

        if args.preemptible:
            assert args.autoresume is True, "Preemptible training requires autoresume"
            base_run.scheduling = {"resumable": True, "priority": "low"}
            base_run.parameters["autoresume"] = True
        else:
            base_run.parameters["autoresume"] = args.autoresume

        base_run.parameters["global_seed"] = seed

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            args.model_size,
            f"dataset-{args.dataset}",
            f"domain-{domain_types}",
            f"subsample-dist-{args.subsample_dist}",
            f"samples-{args.num_samples}",
        ]

        base_run.parameters[
            "device_train_microbatch_size"] = args.device_batch_size
        base_run.parameters["device_eval_batch_size"] = args.device_batch_size

        base_run.parameters["train_loader"]["dataset"][
            "streams"] = domain_streams

        if not args.no_log_domain_loss:
            base_run.parameters["callbacks"]["log_domain_loss"] = {
                "num_domains": args.num_domains
            }

        if args.local_debug:
            with open("debug-ref.yaml", "w") as f:
                OmegaConf.save(config=OmegaConf.create(base_run.parameters),
                               f=f)
        else:
            run = create_run(base_run)
            print(
                f"Launched reference training for seed {seed} with in {run.name}"
            )
