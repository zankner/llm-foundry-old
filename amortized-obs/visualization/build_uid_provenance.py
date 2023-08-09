import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import wandb
from streaming import StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from tqdm import tqdm
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')

# Probably can just have a script that ranks and sorts the losses and subsamples, then
# send that to build data trajectory

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]


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
                "uid": batch["ref_loss"][idx].item(),
                "pile_set_name": PILE_DATA_SOURCES[batch["pile_set_name"]]
            }
            yield sample


class UIDDataset(IterableDataset):
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
            yield {
                "uid": sample["uid"],
                "provenance": PILE_DATA_SOURCES.index(sample["pile_set_name"])
            }


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--download_remote", type=str, default="pile")
    parser.add_argument("--download-local", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)

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

    streaming_dataset = StreamingDataset(
        remote=args.download_remote,
        local=args.download_local,
        split="train",
    )
    dataset = UIDDataset(streaming_dataset)
    dataloader = build_dataloader(dataset, 512, args.num_workers)

    num_samples = streaming_dataset.size
    samples = generate_samples(dataloader, truncate_num_samples=None)

    uid_to_provenance = {}
    for step, sample in enumerate(
            tqdm(samples, desc="prune", total=num_samples, leave=True)):
        uid_to_provenance[sample["uid"]] = sample["provenance"]
        if use_wandb and step % 1_000 == 0:
            wandb.log(({'step': step, 'progress': step / num_samples}))

    print("Saving uid and provenance ...")
    with open("uid_to_provenance.pkl", "wb") as f:
        pickle.dump(uid_to_provenance, f)
