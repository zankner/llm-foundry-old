import os
import platform
import pickle
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import wandb
import numpy as np
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


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
            sample = {k: v[idx] for k, v in batch.items()}
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
        for sample in self.dataset:
            tokens = self._read_binary_tokenized_sample(sample)
            token_buffer += tokens
            while len(token_buffer) >= self.max_length:
                concat_sample = token_buffer[:self.max_length]
                token_buffer = token_buffer[
                    self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_sample).tobytes(),
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
                        choices=["2B", "5B", "20B"])
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

    streaming_data = StreamingDataset(remote=args.download_remote,
                                      local=args.local,
                                      split="train",
                                      shuffle=True,
                                      shuffle_seed=args.seed,
                                      shuffle_algo="py1b",
                                      shuffle_block_size=16777216,
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

    columns = {'tokens': 'bytes'}

    if args.holdout_num_tokens == "2B":
        holdout_num_tokens = 2_000_000_000
    elif args.holdout_num_tokens == "5B":
        holdout_num_tokens = 5_000_000_000
    elif args.holdout_num_tokens == "20B":
        holdout_num_tokens = 20_000_000_000
    else:
        raise ValueError(
            f"Invalid holdout num tokens: {args.holdout_num_tokens}")

    holdout_num_samples = (holdout_num_tokens // args.max_length) + 1
    assert holdout_num_samples < streaming_data.size, "Truncate num samples too large"
    print(f"Total holdout samples: {holdout_num_samples}")
    print(
        f"Percent training samples: {1 - (holdout_num_samples / streaming_data.size)}"
    )

    denominator = streaming_data.size * (6212 // 4) // args.max_length

    upload_name = f"{args.holdout_num_tokens}-holdout-tokens-sd-{args.seed}"
    upload_remote = os.path.join(args.upload_base, upload_name)
    holdout_writer = MDSWriter(columns=columns,
                               out=os.path.join(upload_remote, "holdout"),
                               compression="zstd")
    training_writer = MDSWriter(columns=columns,
                                out=os.path.join(upload_remote, "train", "base",
                                                 "train"),
                                compression="zstd")
    for step, sample in enumerate(
            tqdm(samples, desc="concat", total=holdout_num_samples,
                 leave=True)):

        if step <= holdout_num_samples:
            holdout_writer.write(sample)
            if step == holdout_num_samples:
                holdout_writer.finish()
            print("Finished writing holdout set")
        else:
            training_writer.write(sample)

        if use_wandb and step % 1_000 == 0:
            wandb.log(({'step': step, 'progress': step / denominator}))

    training_writer.finish()
