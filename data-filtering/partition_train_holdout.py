import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import wandb
import numpy as np
import torch
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

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
            sample = {k: v[idx] for k, v in batch.items()}
            unique_uids = np.unique(sample["uids"])
            sample["uids"] = unique_uids
            yield sample


class ConcatDomainsTokensDataset(IterableDataset):
    """An IterableDataset that returns concatenated token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'domain_idx': int}

    """

    def __init__(
        self,
        dataset: StreamingDataset,
        max_length: int,
        no_wrap: bool,
    ):
        self.dataset = dataset
        self.max_length = max_length
        self.should_wrap = not no_wrap

    def _read_binary_tokenized_sample(self, sample):
        return np.frombuffer(sample['tokens'], dtype=np.int64).copy().tolist()

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        token_buffer = []
        uid_buffer = []
        for sample in self.dataset:
            tokens = self._read_binary_tokenized_sample(sample)
            uids = [sample["uid"]] * len(tokens)
            token_buffer += tokens
            uid_buffer += uids
            while len(token_buffer) >= self.max_length:
                concat_tokens = token_buffer[:self.max_length]
                concat_uids = uid_buffer[:self.max_length]
                token_buffer = token_buffer[
                    self.max_length:] if self.should_wrap else []
                uid_buffer = uid_buffer[
                    self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_tokens).tobytes(),
                    "uids": np.array(concat_uids, dtype=np.int32)
                }


if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--download-remote", type=str, required=True)
    parser.add_argument("--local", type=str, default="/tmp/excess-loss")
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)

    # Domain args
    parser.add_argument("--holdout-num-tokens",
                        type=str,
                        required=True,
                        choices=["2B", "5B", "20B", "26B", "130B"])
    parser.add_argument("--upstream-batch-size", type=int, default=512)

    # Tokenization args
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-wrap", default=False, action='store_true')

    # Upload args
    parser.add_argument("--upload-base", type=str, required=True)

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

    streaming_data = StreamingDataset(
        remote=args.download_remote,
        local=args.local,
        split="train",
        shuffle=True,
        shuffle_algo="py1s",  # For some weird speed reasons
        shuffle_seed=args.seed,
        predownload=16777216,
        num_canonical_nodes=128)

    data = ConcatDomainsTokensDataset(
        streaming_data,
        max_length=args.max_length,
        no_wrap=args.no_wrap,
    )
    loader = build_dataloader(data,
                              batch_size=512,
                              num_workers=args.num_workers)
    samples = generate_samples(loader, truncate_num_samples=None)

    holdout_columns = {"tokens": "bytes", "uids": "ndarray:int32"}
    train_columns = {"tokens": "bytes", "idx": "int", "uids": "ndarray:int32"}

    if args.holdout_num_tokens == "2B":
        holdout_num_tokens = 2_000_000_000
    elif args.holdout_num_tokens == "5B":
        holdout_num_tokens = 5_000_000_000
    elif args.holdout_num_tokens == "20B":
        holdout_num_tokens = 20_000_000_000
    elif args.holdout_num_tokens == "26B":
        holdout_num_tokens = 26_000_000_000
    else:
        raise ValueError(
            f"Invalid holdout num tokens: {args.holdout_num_tokens}")

    holdout_num_samples = -(-holdout_num_tokens // args.max_length)
    assert holdout_num_samples < streaming_data.size, "Truncate num samples too large"
    print(f"Total holdout samples: {holdout_num_samples}")
    print(
        f"Percent training samples: {1 - (holdout_num_samples / streaming_data.size)}"
    )

    upload_name = f"{args.holdout_num_tokens}-holdout-tokens"
    upload_remote = os.path.join(args.upload_base, upload_name)

    holdout_writer = MDSWriter(
        columns=holdout_columns,
        out=os.path.join(upload_remote, "holdout", "train"),
        compression=None,
        max_workers=None,
    )
    training_writer = MDSWriter(columns=train_columns,
                                out=os.path.join(upload_remote, "train", "base",
                                                 "train"),
                                compression="zstd",
                                max_workers=args.num_workers)
    train_idx = 0
    for step, sample in enumerate(
            tqdm(samples, desc="concat", total=streaming_data.size,
                 leave=True)):

        if step <= holdout_num_samples:
            holdout_writer.write(sample)
            if step == holdout_num_samples:
                holdout_writer.finish()
                print("Finished writing holdout set")
        else:
            sample = {**sample, "idx": train_idx}
            training_writer.write(sample)
            train_idx += 1

        if use_wandb and step % 1_000 == 0:
            wandb.log(({'step': step, 'progress': step / streaming_data.size}))

    training_writer.finish()