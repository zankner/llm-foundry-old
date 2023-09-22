import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import wandb
from streaming import StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from tqdm import tqdm
import numpy as np
import pickle

from pretrain_utils import build_ref_base, build_remote_base

torch.multiprocessing.set_sharing_strategy('file_system')

# Probably can just have a script that ranks and sorts the losses and subsamples, then
# send that to build data trajectory


def build_dataloader(dataset, batch_size, num_workers) -> DataLoader:
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = num_workers  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else None

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
        loader: DataLoader,
        eos_token_id: int,
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
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            concat_tokens = batch["tokens"][idx][0].tolist()
            seq_split_tokens = [[]]
            for token in concat_tokens:
                if token == eos_token_id:
                    seq_split_tokens += []
                else:
                    seq_split_tokens[-1].append(str(token))
            seq_split_tokens = [split for split in seq_split_tokens if len(split) != 0]
            for sequence in seq_split_tokens:
                if truncate_num_samples is not None and n_samples == truncate_num_samples:
                    return
                n_samples += 1
                sample = {"tokens": " ".join(sequence)}
                yield sample

class TokensDataset(IterableDataset):
    """An IterableDataset that returns concatenated token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'domain_idx': int}

    """
    def __init__(
        self,
        dataset: StreamingDataset,
    ):
        self.dataset = dataset

    def _read_binary_tokenized_sample(self, sample):
        return np.frombuffer(sample['tokens'], dtype=np.int64).copy().reshape(1, -1)

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.dataset:
            s = {"tokens": self._read_binary_tokenized_sample(sample)}
            yield s

if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--eos-token-id", type=int, default=0)

    # Holdout args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "130B"])
    
    parser.add_argument("--output-file", type=str, required=True)

    # Misc
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None, "Wandb name necessary if using for logging"

        wandb.init(name=args.wandb_name,
                   project="pre-process-data",
                   entity="mosaic-ml")

    remote_base = build_remote_base(
        num_holdout_tokens=args.holdout_num_tokens,
        dataset=args.dataset,
    )
    data_remote = os.path.join(
        remote_base, "holdout"
    )

    streaming_dataset = StreamingDataset(
        remote=data_remote,
        local=args.download_local,
        split="train",
    )
    dataset = TokensDataset(streaming_dataset)
    dataloader = build_dataloader(dataset, 512, args.num_workers)

    samples = generate_samples(dataloader, eos_token_id=args.eos_token_id, truncate_num_samples=None)

    with open(args.output_file, "w") as f:
        for step, sample in enumerate(
                tqdm(samples, desc="prune", total=streaming_dataset.size, leave=True)):
            f.write(sample["tokens"] + "\n")
    
