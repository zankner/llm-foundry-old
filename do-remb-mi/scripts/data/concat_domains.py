import os
import tempfile
import pickle
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

import torch
import wandb
import numpy as np
from streaming import MDSWriter, StreamingDataset
from composer.utils import get_file
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from utils import get_sample_int_keys

SHUFFLE_SEED = 17  # Fixing so that domains are properly mixed
torch.multiprocessing.set_sharing_strategy('file_system')

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]


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
            sample = get_sample_int_keys({
                k: v[idx] for k, v in batch.items()
            },
                                         int_keys=["domain_idx"])
            yield sample, sample["domain_idx"]


class ConcatDomainsTokensDataset(IterableDataset):
    """An IterableDataset that returns concatenated token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'domain_idx': int}

    """

    def __init__(self,
                 dataset: StreamingDataset,
                 num_domains: int,
                 domain_source: str,
                 max_length: int,
                 no_wrap: bool,
                 uid_to_domain_path: str = None):
        self.dataset = dataset
        self.num_domains = num_domains
        self.domain_source = domain_source
        self.max_length = max_length
        self.should_wrap = not no_wrap

        # Building uid to domain mapping
        if self.domain_source == "data-source":
            self.uid_to_domain = PILE_DATA_SOURCES
        elif self.domain_source == "clusters":
            assert uid_to_domain_path is not None, "uid_to_domain_path must be provided when using clusters"
            with tempfile.NamedTemporaryFile() as tmp_file:
                get_file(uid_to_domain_path,
                         tmp_file.name,
                         overwrite=True,
                         progress_bar=True)

                with open(tmp_file.name, "rb") as f:  # Feels hacky but oh well
                    self.uid_to_domain = pickle.load(f)

    def _get_domain(self, sample) -> int:
        if self.domain_source == "data-source":
            return self.uid_to_domain.index(sample["pile_set_name"])
        elif self.domain_source == "clusters":
            return self.uid_to_domain[sample["uid"]]
        else:
            raise ValueError(f"Unsupported domain source: {self.domain_source}")

    def _read_binary_tokenized_sample(self, sample):
        return np.frombuffer(sample['tokens'], dtype=np.int64).copy().tolist()

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffers = [[] for _ in range(self.num_domains)]
        for sample in self.dataset:
            domain_idx = self._get_domain(sample)
            tokens = self._read_binary_tokenized_sample(sample)
            buffers[domain_idx] = buffers[domain_idx] + tokens
            while len(buffers[domain_idx]) >= self.max_length:
                concat_sample = buffers[domain_idx][:self.max_length]
                buffers[domain_idx] = buffers[domain_idx][
                    self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_sample).tobytes(),
                    "domain_idx": domain_idx,
                }


# TODO: REFAC TO TAKE IN THE EMBEDDINGS FOR CLUSTERS
if __name__ == "__main__":

    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--download-remote", type=str, required=True)
    parser.add_argument("--local", type=str, default="/tmp/sample-domains")
    parser.add_argument("--splits",
                        type=str,
                        nargs="+",
                        default=["train", "val", "test"])

    # Domain args
    parser.add_argument("--truncate-num-samples", type=int, default=None)
    parser.add_argument("--upstream-batch-size", type=int, default=512)
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--domain-source",
                        type=str,
                        required=True,
                        choices=["data-source", "clusters"])
    parser.add_argument("--uid-to-domain-path", type=str, default=None)

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

    if args.truncate_num_samples is not None:
        truncate_num_samples = args.truncate_num_samples * args.upstream_batch_size
    else:
        truncate_num_samples = None

    for split in args.splits:
        print(f"Converting split {split}")
        streaming_data = StreamingDataset(remote=args.download_remote,
                                          local=args.local,
                                          split=split,
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
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader,
                                   truncate_num_samples=truncate_num_samples)

        columns = {'tokens': 'bytes', 'domain_idx': 'int'}
        if truncate_num_samples is not None:
            denominator = truncate_num_samples
        else:
            denominator = streaming_data.size * (
                6212 //
                4) // args.max_length  # Estimate tokens / tokens per sample

        writers = [
            MDSWriter(columns=columns,
                      out=os.path.join(
                          f"{args.upload_remote}-sd-{SHUFFLE_SEED}",
                          f"domain-{domain_idx}", split),
                      compression="zstd")
            for domain_idx in range(args.num_domains)
        ]
        for step, (sample, domain_idx) in enumerate(
                tqdm(samples, desc=split, total=denominator, leave=True)):

            writers[domain_idx].write(sample)

            if use_wandb and step % 1_000 == 0:
                wandb.log(({'step': step, 'progress': step / denominator}))

        for writer in writers:
            writer.finish()
