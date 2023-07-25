def set_common_args(args, base_run, run_name, seed):
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
        f"model-size-{args.model_size}", f"seed-{seed}", f"packing-{packing}"
    ]

    # Handle preemption
    if args.preemptible:
        assert args.autoresume is True, "Preemptible training requires autoresume"
        base_run.scheduling = {"resumable": True, "priority": "low"}
        base_run.parameters["autoresume"] = True
    else:
        base_run.parameters["autoresume"] = args.autoresume

    # Set batch size
    base_run.parameters["device_train_microbatch_size"] = args.device_batch_size
    base_run.parameters["device_eval_batch_size"] = args.device_batch_size

    # Set training duration
    base_run.parameters["max_duration"] = f"{num_samples}ba"

    # Set eval/ckpt freq
    if num_samples in [25_000]:
        base_run.parameters["eval_interval"] = "1000ba"
        base_run.parameters["save_interval"] = "500ba"
    else:
        raise ValueError(f"Invalid num_samples {num_samples}")
