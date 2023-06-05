import argparse
from mcli import RunConfig, create_run

from utils import build_domain_streams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--model-size", type=str, default="125M")
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--num-domains", type=int, default=22)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--num-steps",
                        type=str,
                        required=True,
                        choices=["100K", "1ep"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    args = parser.parse_args()

    domain_streams = build_domain_streams(
        args.num_domains,
        proportions=[1 / args.num_domains for _ in range(args.num_domains)])

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/reference/pretrain_ref.yaml")

        base_run.name = f"zack-ref-{args.model_size}-step-{args.num_steps}-sd-{seed}".lower(
        )
        base_run.parameters[
            "run_name"] = f"ref-{args.model_size}-param-step-{args.num_steps}"

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
            args.model_size, f"steps-{args.num_steps}"
        ]

        base_run.parameters[
            "device_train_microbatch_size"] = args.device_batch_size
        base_run.parameters[
            "device_eval_microbatch_size"] = args.device_batch_size

        base_run.parameters["train_loader"]["dataset"][
            "streams"] = domain_streams

        if args.num_steps == "100K":
            base_run.parameters["max_duration"] = "100000ba"
        elif args.num_steps == "1ep":
            base_run.parameters["max_duration"] = "1ep"

        run = create_run(base_run)
        print(f"Launched reference training for seed {seed} with in {run.name}")