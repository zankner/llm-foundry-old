import argparse
import os

from omegaconf import OmegaConf
from mcli import RunConfig, create_run

if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=64)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--no-packing", action="store_true")

    # Data args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--tokenizer-prefix",
                        type=str,
                        default="gpt-neox-20b-seqlen-2048")
    parser.add_argument("--concat-method",
                        type=str,
                        required=True,
                        choices=["random", "data-source", "clusters"])
    parser.add_argument("--num-clusters", type=int, default=None)

    args = parser.parse_args()

    if args.concat_method == "clusters":
        assert args.num_clusters is not None, "Must specify num clusters if using cluster concat"

    packing = not args.no_packing
    if args.concat_method == "random" and not packing:
        run_type = "random-masked"
    elif args.concat_method == "random" and packing:
        run_type = "random-packing"
    elif args.concat_method == "data-source" and packing:
        run_type = "data-source-packing"
    elif args.concat_method == "clusters" and packing:
        run_type = f"{args.num_clusters}-clusters-packing"
    else:
        raise ValueError(
            f"Invalid concat and packing combination: concat={args.concat_method}, packing={packing}"
        )

    run_name = f"{run_type}-ds-{args.dataset}-np-{args.model_size}"
    if args.concat_method == "clusters":
        concat_prefix = f"{args.num_clusters}-clusters"
    else:
        concat_prefix = args.concat_method
    data_remote = os.path.join("oci://mosaicml-internal-sem-concat",
                               args.dataset, "pre-concat",
                               args.tokenizer_prefix, concat_prefix)

    if args.model_size == "125M":
        num_samples = 25_000
        model_cfg = {"d_model": 768, "n_heads": 12, "n_layers": 12}
    elif args.model_size == "250M":
        num_samples = 25_000
        model_cfg = {"d_model": 1024, "n_heads": 16, "n_layers": 16}
    elif args.model_size == "1B":
        num_samples = 25_000
        model_cfg = {"d_model": 2048, "n_heads": 16, "n_layers": 24}
    else:
        raise ValueError(f"Unknown model size {args.model_size}")

    for seed in args.seeds:
        base_run = RunConfig.from_file(f"sem-concat/yamls/models/pretrain.yaml")

        # Set run name
        base_run.name = run_name.lower()[:56]  # Mcli things
        base_run.parameters["run_name"] = run_name

        # Set seed
        base_run.parameters["global_seed"] = seed

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

        # Set modeling args
        base_run.parameters["model"]["d_model"] = model_cfg["d_model"]
        base_run.parameters["model"]["n_heads"] = model_cfg["n_heads"]
        base_run.parameters["model"]["n_layers"] = model_cfg["n_layers"]

        # Set attn config
        base_run.parameters["model"]["attn_config"][
            "attn_uses_sequence_id"] = not packing

        # Set data args
        base_run.parameters["train_loader"]["dataset"]["remote"] = data_remote

        # Common wandb tags
        base_run.parameters["loggers"]["wandb"]["tags"] += [
            f"dataset-{args.dataset}", f"concat-{concat_prefix}",
            f"model-size-{args.model_size}", f"seed-{seed}",
            f"packing-{packing}"
        ]

        # Handle preemption
        if args.preemptible:
            assert args.autoresume is True, "Preemptible training requires autoresume"
            base_run.scheduling = {"resumable": True, "priority": "low"}
            base_run.parameters["autoresume"] = True
        else:
            base_run.parameters["autoresume"] = args.autoresume

        # Set batch size
        base_run.parameters[
            "device_train_microbatch_size"] = args.device_batch_size
        base_run.parameters["device_eval_batch_size"] = args.device_batch_size

        # Set training duration
        base_run.parameters["max_duration"] = f"{num_samples}ba"

        # Set eval/ckpt freq
        if num_samples in [25_000]:
            base_run.parameters["eval_interval"] = "1000ba"
            base_run.parameters["save_interval"] = "500ba"
        else:
            raise ValueError(f"Invalid num_samples {num_samples}")

        if args.local_debug:
            with open("debug.yaml", "w") as f:
                OmegaConf.save(config=OmegaConf.create(base_run.parameters),
                               f=f)
        else:
            run = create_run(base_run)
            print(f"Launched seed {seed} with in {run.name}")