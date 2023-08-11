import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import wandb
from streaming import MDSWriter, StreamingDataset
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from pretrain_utils import build_remote_base

torch.multiprocessing.set_sharing_strategy('file_system')


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
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            sample = {
                "tokens": batch["tokens"][idx],
                "idx": batch["idx"][idx].item()
            }
            yield sample


class RefLossDataset(IterableDataset):
    """An IterableDataset that returns concatenated token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'domain_idx': int}

    """

    def __init__(
        self,
        dataset: StreamingDataset,
    ):
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.dataset:
            yield {"tokens": sample["tokens"], "idx": sample["idx"]}


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--download-local", type=str, default="/tmp/download")
    parser.add_argument("--num-workers", type=int, default=64)

    # Holdout args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Final args
    parser.add_argument("--available-num-tokens",
                        type=str,
                        required=True,
                        choices=["13B", "26B"])

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

    download_base = build_remote_base(
        dataset=args.dataset,
        num_tokens=args.holdout_num_tokens,
    )
    download_remote = os.path.join(download_base, "train", "base")
    streaming_data = StreamingDataset(
        remote=download_remote,
        local=args.download_local,
        split="train",
        shuffle=False,
    )

    if args.available_num_tokens == "13B":
        available_num_tokens = 13_000_000_000
    elif args.available_num_tokens == "26B":
        available_num_tokens = 26_000_000_000
    else:
        raise ValueError(
            f"Invalid holdout num tokens: {args.available_num_tokens}")
    num_samples = int(-(-available_num_tokens // 2048))

    samples = generate_samples(streaming_data, truncate_num_samples=num_samples)
    columns = {'tokens': 'bytes', 'idx': 'int'}

    upload_base = build_remote_base(num_holdout_tokens=args.holdout_num_tokens,
                                    dataset=args.dataset)
    upload_remote = os.path.join(
        upload_base, f"{args.available_num_tokens}-available-tokens", "train")

    # print(f"Uploading {num_samples} samples to {upload_remote}")
    # with MDSWriter(
    #         columns=columns,
    #         out=os.path.join(upload_remote, "train"),
    #         compression="zstd",
    #         max_workers=args.num_workers,
    # ) as streaming_writer:
    #     for step, sample in enumerate(
    #             tqdm(samples, desc="subsample", total=num_samples, leave=True)):
    #         streaming_writer.write(sample)
    #         if use_wandb and step % 1_000 == 0:
    #             wandb.log(({'step': step, 'progress': step / num_samples}))
