import os
import tempfile
import pickle
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable, List

import wandb
from streaming import MDSWriter, StreamingDataset
import torch
from tqdm import tqdm
from composer.utils import get_file

from pretrain_utils import (CKPT_BASE, build_ref_base, build_remote_base)

torch.multiprocessing.set_sharing_strategy('file_system')


def generate_samples(
        dataset: StreamingDataset,
        data_trajectory: List,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for sample_idx in data_trajectory:
        if truncate_num_samples is not None and n_samples == truncate_num_samples:
            return
        n_samples += 1
        sample = dataset[sample_idx]
        sample = {
            "tokens": sample["tokens"],
            "idx": sample["idx"],
        }
        if sample_idx != sample["idx"]:
            raise ValueError("All sample indices should be the same")
        yield sample


def fetch_data_trajectory(sample_path: str):
    # Loading the data trajectory
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(sample_path, tmp_file.name, overwrite=True)
        with open(tmp_file.name, "rb") as f:
            joint_loss_idx = pickle.load(f)
    data_trajectory = joint_loss_idx["indices"] 
    return data_trajectory


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)

    # Holdout args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])
    
    # Final args
    parser.add_argument("--final-num-tokens", type=str, required=True, choices=["2B", "5B", "20B", "26B", "130B"])

    # Selection method
    parser.add_argument("--score-method", type=str, required=True, choices=["llm", "ngram"])
    parser.add_argument("--mine-type", type=str, required=True, choices=["easy", "mid", "hard"])
    parser.add_argument("--reduction-rate", type=float, required=True, choices=[0.5, 0.25, 0.125])

    # Misc
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None, "Wandb name necessary if using for logging"

        wandb.init(name=args.wandb_name,
                   project="pre-process-data",
                   entity="mosaic-ml")

    offline_base = f"{args.final_num_tokens}-final-tokens-{args.mine_type}-mine-reduction-{args.reduction_rate}"
    # Building proxy run name
    if args.score_method == "llm":
        assert args.ref_num_tokens is not None and args.ref_model_size is not None
        
        ref_run_base = build_ref_base(args.ref_num_tokens, args.ref_model_size)
        run_name = f"ref-{args.dataset}-{ref_run_base}-holdt-{args.holdout_num_tokens}"

        offline_samples_path = os.path.join(CKPT_BASE, args.dataset, "reference",
                                f"{run_name}-sd-{args.seed}", offline_base,
                                "joint_loss_idx.pkl")
        
        upload_name = f"{args.mine_type}-mine-reduction-{args.reduction_rate}-{ref_run_base}"
    else:
        upload_name = f"{args.mine_type}-mine-reduction-{args.reduction_rate}-ngram"
        pass

    print(f"Loading samples from {offline_samples_path}")
    data_trajectory = fetch_data_trajectory(offline_samples_path)
    num_samples = len(data_trajectory)
    print(f"Number of subsampled documents: {num_samples}")

    streaming_data = StreamingDataset(
        local=args.download_local,
        split="train",
        shuffle=False,
    )

    samples = generate_samples(streaming_data,
                               data_trajectory=data_trajectory,
                               truncate_num_samples=None)
    columns = {'tokens': 'bytes', 'idx': 'int'}

    upload_base = build_remote_base(num_holdout_tokens=args.holdout_num_tokens,
                                    dataset=args.dataset)
    upload_remote = os.path.join(
        upload_base, "pruned",
        f"{args.final_num_tokens}-final-tokens-pruned-from-offline-{upload_name}-sd-{args.seed}"
    )

    print(f"Uploading samples to {upload_remote}")
    with MDSWriter(
            columns=columns,
            out=os.path.join(upload_remote, "train"),
            compression="zstd",
            max_workers=args.num_workers,
    ) as streaming_writer:
        for step, sample in enumerate(
                tqdm(samples, desc="prune", total=num_samples, leave=True)):
            streaming_writer.write(sample)
            if use_wandb and step % 1_000 == 0:
                wandb.log(({'step': step, 'progress': step / num_samples}))
