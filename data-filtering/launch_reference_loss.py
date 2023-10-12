import argparse

from omegaconf import OmegaConf as om
from mcli import RunConfig, create_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--ref-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])

    # Final args
    parser.add_argument("--final-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])

    # Data args
    parser.add_argument("--device-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer", type=str, default="gpt4-tiktoken")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dataset",
                        type=str,
                        default="mpt",
                        choices=["mpt", "pile"])
    parser.add_argument("--num-passes", type=str, required=True)
    args = parser.parse_args()

    base_run = RunConfig.from_file(
        f"data-filtering/yamls/build_reference_loss.yaml")

    # Set compute
    base_run.cluster = args.cluster
    base_run.gpu_num = args.ngpus
    base_run.image = "mosaicml/llm-foundry:2.1.0_cu121-e772a47"

    # Set rest of cluster params
    if args.cluster in ["r9z1", "r14z3"]:
        base_run.gpu_type = "h100_80gb"
    elif args.cluster in ["r8z6", "r1z1"]:
        base_run.gpu_type = "a100_80gb"
    else:
        base_run.gpu_type = "a100_40gb"

    # Set ref model args
    base_run.command = base_run.command.replace(r"{ref_model_size}",
                                                str(args.ref_model_size))
    base_run.command = base_run.command.replace(r"{ref_num_tokens}",
                                                str(args.ref_num_tokens))
    base_run.command = base_run.command.replace(r"{seed}", str(args.seed))

    # Set final args
    base_run.command = base_run.command.replace(r"{final_num_tokens}",
                                                str(args.final_num_tokens))

    # Set dataset args
    base_run.command = base_run.command.replace(r"{tokenizer}",
                                                str(args.tokenizer))
    base_run.command = base_run.command.replace(r"{dataset}", str(args.dataset))
    base_run.command = base_run.command.replace(r"{num_passes}",
                                                str(args.num_passes))
    base_run.command = base_run.command.replace(r"{device_batch_size}",
                                                str(args.device_batch_size))

    # Set run name
    run_name = f"sd-{args.seed}-build-ref-loss"
    base_run.run_name = run_name
    base_run.name = run_name

    if args.local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(base_run), f=f)
    else:
        run = create_run(base_run)
        print(f"Launched seed {args.seed} with in {run.name}")