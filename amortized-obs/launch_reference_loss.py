import os
import argparse

from omegaconf import OmegaConf as om
from mcli import RunConfig, create_run

from pretrain_utils import (CKPT_BASE, build_ref_base, build_remote_base,
                            launch_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=128)
    parser.add_argument("--ref-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "26B"])
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "26B"])
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--local-debug", action="store_true")
    args = parser.parse_args()

    base_run = RunConfig.from_file(
        f"amortized-obs/yamls/build_reference_loss.yaml")

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

    remote_base = build_remote_base(
        num_holdout_tokens=args.holdout_num_tokens,
        dataset=args.dataset,
    )
    # Set remote source dataset
    download_remote = os.path.join(remote_base, "train", "base")
    base_run.command = base_run.command.replace(r"{download_remote}",
                                                download_remote)

    # Set remote upload dataset
    ref_base = build_ref_base(args.ref_num_tokens, args.ref_model_size)
    upload_remote = os.path.join(remote_base, "train",
                                 f"{ref_base}-sd-{args.seed}", "train")
    base_run.command = base_run.command.replace(r"{upload_remote}",
                                                upload_remote)

    # Set batch information
    base_run.command = base_run.command.replace(r"{device_batch_size}",
                                                str(args.device_batch_size))

    # Set model information
    ref_run_name = f"ref-{args.dataset}-{ref_base}-holdt-{args.holdout_num_tokens}"
    model_ckpt = os.path.join(CKPT_BASE, args.dataset, "reference",
                              f"{ref_run_name}-sd-{args.seed}", "ckpts",
                              "latest-rank0.pt.symlink")

    base_run.command = base_run.command.replace(r"{model_ckpt}", model_ckpt)
    base_run.command = base_run.command.replace(r"{model_size}",
                                                args.ref_model_size)

    # Set run name
    run_name = f"sd-{args.seed}-build-ref-loss-{ref_base}"
    base_run.run_name = run_name
    base_run.name = run_name

    if args.local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(base_run), f=f)
    else:
        run = create_run(base_run)
        print(f"Launched seed {args.seed} with in {run.name}")