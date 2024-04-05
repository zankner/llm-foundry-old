import os
import re
import argparse

from mcli import RunConfig, create_run
from omegaconf import OmegaConf as om

def build_model_cfg(step, model_size):
    if step == "final":
        step_fmt = "latest-rank0.pt.symlink"
    else:
        step_fmt = f"ba{step}/rank0.pt"
    model_cfg = {
        "model_name": "${run_name}",  # Set in launch script
        "load_path": "oci://mosaicml-internal-checkpoints/zack/me-fomo-data-filtering/${dataset}/gpt4-tiktoken-seqlen-${max_seq_len}/final/${ckpt_fmt_run_name}-bs-${bs}-lr-${lr}-sd-${seed}/ckpts/" + step_fmt,
        "model": {
            "name": "mpt_causal_lm",
            "init_device": "meta",
            "d_model": 2048,  # Change for 3b runs
            "n_heads": 16,
            "n_layers": 24,
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
        },
        "tokenizer": {
            "name": "${tokenizer_name}",  # default tokenizer used for MPT
            "kwargs": {
                "model_name": "gpt-4"
            }
        }
    }

    if model_size == "1B":
        model_cfg["model"]["d_model"] = 2048
        model_cfg["model"]["n_heads"] = 16
        model_cfg["model"]["n_layers"] = 24
    elif model_size == "3B":
        model_cfg["model"]["d_model"] = 2560
        model_cfg["model"]["n_heads"] = 32
        model_cfg["model"]["n_layers"] = 32

    return model_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r15z1")
    parser.add_argument("--ngpus", type=str, default=8)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--eval-freq", type=float, default=0.1)
    parser.add_argument("--priority", type=str, default="low")
    parser.add_argument("--not-preemptible", action="store_true")
    parser.add_argument("--intermediate-eval", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    for seed in args.seeds:
        base_run = RunConfig.from_file(f"data-filtering/yamls/gauntlet.yaml")

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
        if training_duration == "26B":
            training_duration = 26 * 1e+9
        elif training_duration == "52B":
            training_duration = 52 * 1e+9
        
        ckpt_freq = 500 # Assuming that this is the ckpt_freq for all models rn
        token_per_batch = batch_size * seq_len
        all_model_cfgs = []
        if args.intermediate_eval:
            for eval_idx in range(1, int(1 / args.eval_freq)):
                eval_percent = args.eval_freq * eval_idx
                eval_token = eval_percent * training_duration
                eval_batch = int(eval_token / token_per_batch / ckpt_freq) * ckpt_freq
                model_cfg = build_model_cfg(eval_batch, num_params) 
                all_model_cfgs.append(model_cfg)
        else:
            all_model_cfgs.append(build_model_cfg("final", num_params))

        for model_cfg in all_model_cfgs:
            base_run.parameters["models"] = [model_cfg]

            if args.debug:
                with open("debug.yaml", "w") as f:
                            f.write(str(base_run))
            else:
                launched_run = create_run(base_run)
                print(f"Launched run: {launched_run.name}")