import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset


def build_dataloader(dataset, batch_size) -> DataLoader:
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = 64  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

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
            yield {k: v[idx] for k, v in batch.items()}


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.dataset:
            # print(sample)
            # convert to bytes to store in MDS binary format
            yield {
                'text': sample['text'],
                'pile_set_name': sample['pile_set_name']
            }


if __name__ == "__main__":
    s3_remote = "s3://mosaicml-internal-dataset-the-pile/mds/2"
    local = "/tmp/base"
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"Converting split {split}")
        s3_data = StreamingDataset(remote=s3_remote,
                                   local=local,
                                   split=split,
                                   shuffle=False)
        data = NoConcatDataset(s3_data)
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {'text': 'str'}
        denominator = 210607728

        with MDSWriter(columns=columns,
                       out=os.path.join("data", "base", split),
                       compression="zstd") as out:
            for sample in tqdm(samples,
                               desc=split,
                               total=denominator,
                               leave=True):
                out.write(sample)
