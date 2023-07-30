from launch_seed import run_seed
from launch_trees import run_trees
from mcli import wait_for_run_status

def run_bbb(args):
    for sd in args.seeds:
        seed_run = run_seed(args, sd)
        wait_for_run_status(seed_run, 'COMPLETED')
        
        tree_runs = run_trees(args, sd)
        for run in tree_runs:
            wait_for_run_status(seed_run, 'COMPLETED')

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

    args = parser.parse_args()

    run_bbb(args)
