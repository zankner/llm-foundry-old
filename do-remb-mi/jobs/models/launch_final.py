import argparse
from mcli import RunConfig, create_run

from utils import build_domain_streams

baseline_proportions = [
    0.16, 0.127, 0.114, 0.09, 0.102, 0.098, 0.055, 0.056, 0.029, 0.024, 0.021,
    0.017, 0.037, 0.021, 0.012, 0.007, 0.009, 0.006, 0.008, 0.004, 0.002, 0.001
]


def get_proportions(domain_source: str):
    if args.domain_source == "baseline":
        return baseline_proportions
    else:
        raise ValueError(f"Unknown domain source {domain_source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r8z6")
    parser.add_argument("--ngpus", type=int, default=32)
    parser.add_argument("--model-size", type=str, default="1B")
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--num-domains", type=int, default=22)
    parser.add_argument("--domain-source", type=str, required=True)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--num-steps",
                        type=str,
                        required=True,
                        choices=["100K", "1ep"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    args = parser.parse_args()

    proportions = get_proportions(args.domain_source)
    domain_streams = build_domain_streams(args.num_domains,
                                          proportions=proportions)

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/final/pretrain_final.yaml")

        base_run.name = f"zack-final-{args.model_size}-step-{args.num_steps}-sd-{seed}".lower(
        )
        base_run.parameters[
            "run_name"] = f"final-{args.model_size}-param-step-{args.num_steps}-ds-{args.domain_source}"

        if args.preemptible:
            assert args.autoresume is True, "Preemptible training requires autoresume"
            base_run.scheduling = {"resumable": True, "priority": "low"}
            base_run.parameters["autoresume"] = True
        else:
            base_run.parameters["autoresume"] = args.autoresume

        base_run.parameters["global_seed"] = seed

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            args.model_size, f"steps-{args.num_steps}",
            f"ds-{args.domain_source}"
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
        print(f"Launched final training for seed {seed} with in {run.name}")