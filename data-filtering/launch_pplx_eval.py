import re
import argparse

from mcli import RunConfig, create_run

def build_model_cfg(model_size):
    model_cfg = {
        "name": "mpt_causal_lm",
        "init_device": "meta",
        "d_model": None,  # Change for 3b runs
        "n_heads": None,
        "n_layers": None,
        "expansion_ratio": 4,
        "max_seq_len": "${max_seq_len}",
        "vocab_size": 100352,  # update for hero run with custom tokenizer
        "tokenizer_name": "${tokenizer_name}",
        "no_bias": True,
        "norm_type": "low_precision_layernorm",
        "emb_pdrop": 0,
        "resid_pdrop": 0,
        "init_config": {
            "init_nonlinearity": "relu",
            "name": "kaiming_normal_"
        },
        "attn_config": {
            "alibi": True,
            "attn_impl": "triton",
            "clip_qkv": 6,
            "attn_uses_sequence_id": False,
            "attn_pdrop": 0
        }
    }

    if model_size == "1B":
        model_cfg["d_model"] = 2048
        model_cfg["n_heads"] = 16
        model_cfg["n_layers"] = 24
    elif model_size == "3B":
        model_cfg["d_model"] = 2560
        model_cfg["n_heads"] = 32
        model_cfg["n_layers"] = 32

    return model_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--ngpus", type=str, default=8)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--priority", type=str, default="low")
    parser.add_argument("--not-preemptible", action="store_true")

    args = parser.parse_args()

    for seed in args.seeds:
        base_run = RunConfig.from_file(f"data-filtering/yamls/pplx_eval.yaml")

        training_details = re.search(r"final-\d+B-\d+B", args.run_name).group().split("-")
        num_params = training_details[1]
        training_duration = training_details[2]
        dataset = args.run_name.split("-")[0]

        # Set hparams
        batch_size = 512 # Assuming that this is the bs for all models rn
        seq_len = 2048 # Assuming that this is the seq_len for all models rn
        if num_params == "1B":
            lr = 0.0002 # Assuming that this is the lr for all models rn
        elif num_params == "3B":
            lr = 0.00016
        ref_batch_size = 512
        ref_lr = 0.0002

        ckpt_fmt_run_name = args.run_name
        if "global" in ckpt_fmt_run_name:
            ckpt_fmt_run_name += f"-bs-{ref_batch_size}-lr-{ref_lr}"

        # Set name
        base_run.name = (f"sd-{seed}-" + args.run_name)[:56]
        base_run.name = base_run.name.replace(".", "-")
        if base_run.name[-1] == "-":
            base_run.name = base_run.name[:-1]
        base_run.parameters["run_name"] = args.run_name
        base_run.parameters["ckpt_fmt_run_name"] = ckpt_fmt_run_name

        # Set scheduling
        base_run.scheduling["priority"] = args.priority
        base_run.scheduling["preemptible"] = not args.not_preemptible

        # Set hparams
        base_run.parameters["dataset"] = dataset
        base_run.parameters["bs"] = batch_size
        base_run.parameters["lr"] = lr
        base_run.parameters["max_seq_len"] = seq_len


        # Set seed for reasons
        base_run.parameters["seed"] = seed

        # Set compute
        base_run.compute["cluster"] = args.cluster
        base_run.compute["gpus"] = args.ngpus

        # Set batch information
        base_run.parameters["device_eval_batch_size"] = args.device_batch_size

        # Build all model eval ckpts
        base_run.parameters["model"] = build_model_cfg(num_params)

        # Set the eval dataset
        tokenizer_name = "gpt4-tiktoken" # Hard fix for now
        # eval_dataset_remote = f"s3://data-force-one-datasets/__unitystorage/catalogs/36798a58-e180-4029-8cd7-842e61841ef0/volumes/b9e4994e-997d-4cbf-b76b-e38ff5533785/{dataset}/{tokenizer_name}-seqlen-{seq_len}/26B-total-available-holdout-tokens-partition-sd-17/train/26B-tokens-from-0.25-passes/combined/mds"
        eval_dataset_remote = f"s3://data-force-one-datasets/__unitystorage/catalogs/36798a58-e180-4029-8cd7-842e61841ef0/volumes/b9e4994e-997d-4cbf-b76b-e38ff5533785/{dataset}/{tokenizer_name}-seqlen-{seq_len}/base/combined/mds/test"
        base_run.parameters["eval_loader"]["dataset"]["remote"] = eval_dataset_remote

        # with open("debug.yaml", "w") as f:
        #     f.write(str(base_run))

        launched_run = create_run(base_run)
        print(f"Launched run: {launched_run.name}")