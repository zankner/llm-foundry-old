import argparse
import time
from launch_seed import run_seed
from launch_trees import run_trees
from mcli import RunConfig

def run_bbb(args):
    for sd in args.seeds:
        seed_run = run_seed(args, sd)
        while True:
            if seed_run.status == "COMPLETED":
                break
            time.sleep(2)
        
        tree_runs, tree_ckpt_names = run_trees(args, sd)
        num_trees = len(tree_runs)
        num_completed = 0
        for run in tree_runs:
            if run.status == "COMPLETED":
                num_completed += 1
                if num_completed == num_trees:
                    break
            time.sleep(2)

        ckpts = [run.submitted_config.parameters["save_folder"] for run in tree_runs]

        merge_run_config = RunConfig.from_file("brick/yamls/merge_models.yaml")
        merge_run_config.command = merge_run_config.command.replace(
            "${model_ckpts}",
            " ".join(tree_ckpt_names)
        )
        merge_run_config.command = merge_run_config.command.replace(
            "${model_weights}",
            " ".join(args.model_weights)
        )

if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--device-batch-size", type=int, default=32)

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--domain-type",
                        type=str,
                        default="clusters",
                        choices=["clusters", "provenance"])
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])
    parser.add_argument("--warmup-duration",
                        type=float,
                        required=True)
    parser.add_argument("--merge-weights", type=float, nargs="+", default=None)

    args = parser.parse_args()

    run_bbb(args)
