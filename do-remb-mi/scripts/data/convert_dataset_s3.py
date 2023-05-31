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
import wandb

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset


# When creating base version use index as uid for future reference
class StreamingDatasetIndexed(StreamingDataset):

    def __getitem__(self, index: int):
        """Get sample by global index.
        Args:
            index (int): Sample index.
        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard], index


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
            yield {**{k: v[idx] for k, v in batch.items() if k != "uid"}, "uid": batch["uid"][idx].item()}


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes, 'pile_set_name': str, 'uid': int}
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample, idx in self.dataset:
            yield {
                'text': sample['text'],
                'pile_set_name': sample['pile_set_name'],
                'uid': idx.item()
            }


if __name__ == "__main__":
    wandb.init(name="pile-base-convert",
               project="doremi-preprocess",
               entity="mosaic-ml")

    s3_remote = "s3://mosaicml-internal-dataset-the-pile/mds/2"
    local = "/tmp/s3-pile"
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"Converting split {split}")
        s3_data = StreamingDatasetIndexed(remote=s3_remote,
                                          local=local,
                                          split=split,
                                          shuffle=False)
        data = NoConcatDataset(s3_data)
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {'text': 'str', 'pile_set_name': 'str', 'uid': 'int'}
        denominator = 210607728

        with MDSWriter(columns=columns,
                       out=os.path.join("/tmp", "base", split),
                       compression="zstd") as out:
            for step, sample in enumerate(
                    tqdm(samples, desc=split, total=denominator, leave=True)):
                out.write(sample)

                if step % 1_000 == 0:
                    wandb.log(({'step': step, 'progress': step / denominator}))
