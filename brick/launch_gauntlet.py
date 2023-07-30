import os
import argparse

from mcli import RunConfig, create_run
from omegaconf import OmegaConf as om

from pretrain_utils import CKPT_BASE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-ckpt", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--local-debug", action="store_true")
    args = parser.parse_args()

    base_run = RunConfig.from_file(f"rho/yamls/gauntlet.yaml")

    # Set name
    base_run.run_name = f"gauntlet-{args.model_name}"
    base_run.name = f"gauntlet-{args.model_name}"
    base_run.parameters["models"][0]["model_name"] = args.model_name

    # Set compute
    base_run.cluster = args.cluster

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

    # Set batch information
    base_run.parameters["device_eval_batch_size"] = args.batch_size

    # Set model information
    model_ckpt = os.path.join(CKPT_BASE, args.model_ckpt)
    base_run.command = base_run.command.replace(r"{model_ckpt}", model_ckpt)

    if args.local_debug:
        with open("debug.yaml", "w") as f:
            om.save(config=om.create(base_run), f=f)
    else:
        run = create_run(base_run)
        print(f"Created run: {run.name}")