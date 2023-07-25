import os
import argparse

from mcli import RunConfig, create_run

from pretrain_utils import CKPT_BASE, build_ref_base, build_remote_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=128)
    parser.add_argument("--model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B"])
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    base_run = RunConfig.from_file(f"rho/yamls/build_reference_loss.yaml")

    # Set compute
    base_run.cluster = args.cluster
    base_run.gpu_num = args.ngpus

    # Set rest of cluster params
    if args.cluster == "r9z1":
        base_run.image = "mosaicml/llm-foundry:2.0.1_cu118-latest"
        base_run.gpu_type = "h100_80gb"
    elif args.cluster in ["r8z6", "r1z1"]:
        base_run.image = "mosaicml/llm-foundry:1.13.1_cu117-latest"
        base_run.gpu_type = "a100_80gb"
    else:
        base_run.image = "mosaicml/llm-foundry:1.13.1_cu117-latest"
        base_run.gpu_type = "a100_40gb"

    remote_base = build_remote_base(num_holdout_tokens=args.num_tokens,
                                    dataset=args.dataset,
                                    seed=args.seed)
    # Set remote source dataset
    download_remote = os.path.join(remote_base, "train", "base")
    base_run.command = base_run.command.replace(r"{download_remote}",
                                                download_remote)

    # Set remote upload dataset
    ref_base = build_ref_base(args.num_tokens, args.model_size)
    upload_remote = os.path.join(remote_base, "train", ref_base, "train")
    base_run.command = base_run.command.replace(r"{upload_remote}",
                                                upload_remote)

    # Set batch information
    base_run.command = base_run.command.replace(r"{device_batch_size}",
                                                str(args.device_batch_size))

    # Set model information
    ref_run_name = f"ref-{args.dataset}-{ref_base}"
    model_ckpt = os.path.join(CKPT_BASE, "reference",
                              f"{ref_run_name}-sd-{args.seed}", "ckpts",
                              "latest-rank0.pt.symlink")

    base_run.command = base_run.command.replace(r"{model_ckpt}", model_ckpt)
    base_run.command = base_run.command.replace(r"{model_size}",
                                                args.model_size)

    run = create_run(base_run)
    print(f"Created run: {run.name}")