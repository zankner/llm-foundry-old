import argparse
from mcli import RunConfig, create_run

from utils import build_domain_streams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r8z6")
    parser.add_argument("--ngpus", type=int, default=16)
    parser.add_argument("--model-size", type=str, default="125M")
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1e-4)
    parser.add_argument("--init-dist", type="str", default="uniform")
    parser.add_argument("--num-domains", type=int, default=22)
    parser.add_argument("--ref-model", type=str, required=True)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--num-steps",
                        type=str,
                        required=True,
                        choices=["100K", "1ep"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    args = parser.parse_args()

    uniform_proportions = [1 / args.num_domains] * args.num_domains
    domain_streams = build_domain_streams(args.num_domains, uniform_proportions)

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/models/yamls/proxy/pretrain_proxy.yaml")

        base_run.name = f"zack-proxy-{args.model_size}-step-{args.num_steps}-sd-{seed}".lower(
        )
        base_run.parameters[
            "run_name"] = f"proxy-{args.model_size}-param-step-{args.num_steps}-ref-{args.ref_model}"

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
            args.model_size, f"steps-{args.num_steps}", f"ref-{args.ref_model}"
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

        # DomainWeightSetter params
        base_run.parameters["algorithms"]["doremi"][
            "num_domains"] = args.num_domains
        base_run.parameters["algorithms"]["doremi"][
            "step_size"] = args.step_size
        base_run.parameters["algorithms"]["doremi"][
            "smoothing"] = args.smoothing
        base_run.parameters["algorithms"]["doremi"][
            "init_dist"] = args.init_dist

        run = create_run(base_run)
        print(f"Launched proxy training for seed {seed} with in {run.name}")