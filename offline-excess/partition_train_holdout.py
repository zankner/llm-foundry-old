import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import torch
import wandb
import numpy as np
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

SHUFFLE_SEED = 17  # Fixing so that domains are properly mixed
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
        num_domains: int,
        domain_source: str,
        max_length: int,
        no_wrap: bool,
    ):
        self.dataset = dataset
        self.num_domains = num_domains
        self.domain_source = domain_source
        self.max_length = max_length
        self.should_wrap = not no_wrap

    def _read_binary_tokenized_sample(self, sample):
        return np.frombuffer(sample['tokens'], dtype=np.int64).copy().tolist()

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        token_buffer = []
        uid_buffer = []
        num_tokens_buffer = []
        for sample in self.dataset:
            uid = sample["uid"]
            tokens = self._read_binary_tokenized_sample(sample)
            num_tokens = len(tokens)
            token_buffer += tokens
            uid_buffer += [uid] * len(tokens)
            num_tokens_buffer += [num_tokens] * len(tokens)
            while len(token_buffer) >= self.max_length:
                concat_sample = token_buffer[:self.max_length]
                concat_uids = uid_buffer[:self.max_length]
                concat_num_tokens = uid_buffer[:self.max_length]
                token_buffer = token_buffer[
                    self.max_length:] if self.should_wrap else []
                uid_buffer = uid_buffer[
                    self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_sample).tobytes(),
                    "uids": np.asarray(concat_uids).tobytes(),
                    "num_tokens": concat_num_tokens
                }


# TODO: REFAC TO TAKE IN THE EMBEDDINGS FOR CLUSTERS
if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--download-remote", type=str, required=True)
    parser.add_argument("--local", type=str, default="/tmp/sample-domains")
    parser.add_argument("--num-workers", type=int, default=64)

    # Domain args
    parser.add_argument("--truncate-num-samples", type=int, required=True)
    parser.add_argument("--upstream-batch-size", type=int, default=512)

    # Tokenization args
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-wrap", default=False, action='store_true')

    # Upload args
    parser.add_argument("--upload-remote", type=str, required=True)

    # Misc
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None, "Wandb name necessary if using for logging"

        wandb.init(name=args.wandb_name,
                   project="doremi-preprocess",
                   entity="mosaic-ml")

    truncate_num_samples = args.truncate_num_samples * args.upstream_batch_size

    streaming_data = StreamingDataset(remote=args.download_remote,
                                      local=args.local,
                                      split="train",
                                      shuffle=True,
                                      shuffle_seed=SHUFFLE_SEED,
                                      num_canonical_nodes=128)

    data = ConcatDomainsTokensDataset(
        streaming_data,
        num_domains=args.num_domains,
        domain_source=args.domain_source,
        uid_to_domain_path=args.uid_to_domain_path,
        max_length=args.max_length,
        no_wrap=args.no_wrap,
    )
    loader = build_dataloader(data,
                              batch_size=512,
                              num_workers=args.num_workers)
    samples = generate_samples(loader,
                               truncate_num_samples=truncate_num_samples * 2)

    columns = {'tokens': 'bytes', 'uids': 'bytes'}
    denominator = 2 * truncate_num_samples
    assert denominator < data.size, "Truncate num samples too large"

    training_writer = MDSWriter(columns=columns,
                                out=os.path.join(
                                    f"{args.upload_remote}-sd-{SHUFFLE_SEED}",
                                    "train"),
                                compression="zstd")
    holdout_writer = MDSWriter(columns=columns,
                               out=os.path.join(
                                   f"{args.upload_remote}-sd-{SHUFFLE_SEED}",
                                   "holdout"),
                               compression="zstd")
    num_unique = 0
    uid_to_loss_id = {}
    loss_id_to_uid = {}
    num_tokens_per_sample = {}
    for step, (sample, domain_idx) in enumerate(
            tqdm(samples, desc="concat", total=denominator, leave=True)):

        uuids = sorted(set(
            np.frombuffer(sample["uids"], dtype=np.int64).copy().tolist()),
                       key=uuids.index)
        num_tokens = sorted(set(sample["num_tokens"]), key=num_tokens.index)
        del sample["num_tokens"]

        if step <= truncate_num_samples:

            for uid, num_token in zip(uuids, num_tokens):
                if uid not in uid_to_loss_id:
                    uid_to_loss_id[uid] = num_unique
                    loss_id_to_uid[num_unique] = uid
                    num_tokens_per_sample[num_unique] = num_token
                    num_unique += 1

            training_writer.write(sample)
            if step == truncate_num_samples:
                training_writer.finish()
        else:
            holdout_writer.write(sample)

        if use_wandb and step % 1_000 == 0:
            wandb.log(({'step': step, 'progress': step / denominator}))

    holdout_writer.finish()