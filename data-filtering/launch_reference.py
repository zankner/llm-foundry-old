import argparse
import os

from mcli import RunConfig

from pretrain_utils import (CKPT_BASE, set_common_args, launch_run,
                            build_dataset_base)

if __name__ == "__main__":
    # System args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=16)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--priority",
                        type=str,
                        default="lowest",
                        choices=["lowest", "low", "medium"])
    parser.add_argument("--overwrite-shuffle-seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int,
                        required=True)  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Hparams
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--global-batch-size",
                        type=int,
                        default=1024,
                        choices=[512, 1024])

    # Ref Model args
    parser.add_argument("--ref-model-size",
                        type=str,
                        required=True,
                        choices=["125M", "250M", "1B"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "52B", "130B"])

    # Data args
    parser.add_argument("--tokenizer",
                        type=str,
                        default="gpt4-tiktoken",
                        choices=["gpt4-tiktoken", "gpt-neox-20b"])
    parser.add_argument("--available-holdout-tokens",
                        type=str,
                        default="52B",
                        choices=["26B", "52B"])
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="mpt")
    parser.add_argument("--num-passes", type=str, required=True)

    args = parser.parse_args()

    run_name = f"{args.dataset}-passes-{args.num_passes}-ref-{args.ref_model_size}-{args.ref_num_tokens}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"data-filtering/yamls/pretrain_base.yaml")

        data_remote = build_dataset_base(args.dataset,
                                         args.tokenizer,
                                         args.seq_len,
                                         args.ref_num_tokens,
                                         args.num_passes,
                                         args.available_holdout_tokens,
                                         seed=seed,
                                         holdout=True)

        save_base = os.path.join(CKPT_BASE, args.dataset,
                                 f"{args.tokenizer}-seqlen-{args.seq_len}",
                                 "reference")

        set_common_args(args, base_run, run_name, save_base, data_remote,
                        args.ref_model_size, args.ref_num_tokens, seed)

        base_run.parameters["loggers"]["wandb"]["tags"] += [
            "ref", f"ref-params-{args.ref_model_size}",
            f"ref-tok-{args.ref_num_tokens}", f"num-passes-{args.num_passes}",
            f"tokenizer-{args.tokenizer}", f"seq-len-{args.seq_len}"
        ]

        launch_run(base_run, args.local_debug, seed)
