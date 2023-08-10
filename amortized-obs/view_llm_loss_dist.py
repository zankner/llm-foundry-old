import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import wandb
from streaming import StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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
                "ref_loss": batch["ref_loss"][idx].item(),
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
            yield {"ref_loss": sample["ref_loss"], "idx": sample["idx"]}


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--dataset", type=str, default="pile")
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)

    # Holdout args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Reference args
    parser.add_argument("--ref-model-size", type=str, choices=["125M", "250M"])
    parser.add_argument("--ref-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "130B"])

    # Plotting args
    parser.add_argument("--num-samples", type=int, required=True)

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

    ref_run_base = build_ref_base(args.ref_num_tokens, args.ref_model_size)
    remote_base = build_remote_base(
        num_holdout_tokens=args.holdout_num_tokens,
        dataset=args.dataset,
    )
    data_remote = os.path.join(
        remote_base, "train",
        f"{build_ref_base(args.ref_num_tokens, args.ref_model_size)}-sd-{args.seed}"
    )
    print(f"Loading ref losses from {data_remote}")

    streaming_dataset = StreamingDataset(
        remote=data_remote,
        local=args.download_local,
        split="train",
        shuffle=True,
        shuffle_seed=args.seed,
        shuffle_algo="py1b",
    )
    dataset = RefLossDataset(streaming_dataset)
    dataloader = build_dataloader(dataset, 512, args.num_workers)

    samples = generate_samples(dataloader,
                               truncate_num_samples=args.num_samples)

    ref_losses = []
    for step, sample in enumerate(
            tqdm(samples, desc="prune", total=args.num_samples, leave=True)):
        ref_losses.append(sample["ref_loss"])
        if use_wandb and step % 1_000 == 0:
            wandb.log(({'step': step, 'progress': step / args.num_samples}))

    print("Starting sorting of losses ...")
    # Sorting and selecting the losses
    ref_losses = sorted(ref_losses)

    print("Plotting losses")

    sns.kdeplot(ref_losses, shade=True)

    # Vertical lines at the 25%, 50%, and 75% percentiles
    plt.axvline(x=ref_losses[len(ref_losses) // 4],
                color='r',
                linestyle='--',
                label="25% percentile")
    plt.axvline(x=ref_losses[len(ref_losses) // 2],
                color='g',
                linestyle='--',
                label="50% percentile")
    plt.axvline(x=ref_losses[len(ref_losses) * 3 // 4],
                color='b',
                linestyle='--',
                label="75% percentile")

    plt.legend()
    plt.savefig("ref_loss_distribution.png")
