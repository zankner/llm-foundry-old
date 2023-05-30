import os
import platform
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as datasets
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import wandb
import numpy as np

PILE_SUBSETS = {
    "pile-cc": {
        "domain": 0,
        "name": "Pile-CC",
        "raw_samples": 54953117
    },
    "pubmed-central": {
        "domain": 1,
        "name": "PubMed Central",
        "raw_samples": 3098931
    },
    "books3": {
        "domain": 2,
        "name": "Books3",
        "raw_samples": 196640
    },
    "open-web-text-2": {
        "domain": 3,
        "name": "OpenWebText2",
        "raw_samples": 17103059
    },
    "arxiv": {
        "domain": 4,
        "name": "ArXiv",
        "raw_samples": 1264405
    },
    "github": {
        "domain": 5,
        "name": "Github",
        "raw_samples": 19021454
    },
    "free-law": {
        "domain": 6,
        "name": "FreeLaw",
        "raw_samples": 3562015
    },
    "stack-exchange": {
        "domain": 7,
        "name": "Stack Exchange",
        "raw_samples": 15622475
    },
    "uspto-backgrounds": {
        "domain": 8,
        "name": "USPTO Backgrounds",
        "raw_samples": 5883037
    },
    "pubmed-abstracts": {
        "domain": 9,
        "name": "PubMed Abstracts",
        "raw_samples": 15518009
    },
    "gutenberg": {
        "domain": 10,
        "name": "Gutenberg (PG-19)",
        "raw_samples": 28602
    },
    "open-subtitles": {
        "domain": 11,
        "name": "OpenSubtitles",
        "raw_samples": 446612
    },
    "wikepedia-en": {
        "domain": 12,
        "name": "Wikipedia (en)",
        "raw_samples": 6033151
    },
    "dm-mathematics": {
        "domain": 13,
        "name": "DM Mathematics",
        "raw_samples": 1014997
    },
    "ubuntu-irc": {
        "domain": 14,
        "name": "Ubuntu IRC",
        "raw_samples": 10605
    },
    "books-corpus-2": {
        "domain": 15,
        "name": "BookCorpus2",
        "raw_samples": 17868
    },
    "euro-parl": {
        "domain": 16,
        "name": "EuroParl",
        "raw_samples": 69814
    },
    "hacker-news": {
        "domain": 17,
        "name": "HackerNews",
        "raw_samples": 831198
    },
    "youtube-subtitles": {
        "domain": 18,
        "name": "YoutubeSubtitles",
        "raw_samples": 173651
    },
    "phil-papers": {
        "domain": 19,
        "name": "PhilPapers",
        "raw_samples": 33990
    },
    "nih-ex-porter": {
        "domain": 20,
        "name": "NIH ExPorter",
        "raw_samples": 939668
    },
    "enron-emails": {
        "domain": 21,
        "name": "Enron Emails",
        "raw_samples": 517401
    }
}


def _est_progress_denominator(total_samples: int, chars_per_sample: int,
                              chars_per_token: int, max_length: int):
    est_tokens_per_sample = chars_per_sample // chars_per_token
    return total_samples * est_tokens_per_sample // max_length


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


def generate_samples(loader: DataLoader,
                     domain: int,
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
            yield {**{k: v[idx] for k, v in batch.items()}, "domain": domain}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        dataset,
        domain: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.dataset = dataset
        self.domain = domain
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.dataset:
            if self.domain != sample['pile_set_name']:
                continue
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }


if __name__ == "__main__":
    parser = ArgumentParser(description="Splitting pile into named subsets")
    parser.add_argument("--subset", type=str, required=True)
    args = parser.parse_args()

    wandb.init(name=f"pile-{args.subset}-convert",
               project="doremi-preprocess",
               entity="mosaic-ml")

    s3_remote = "s3://mosaicml-internal-dataset-the-pile/mds/2"
    local = "/tmp/s3-pile"
    oci_remote = "oci://mosaicml-internal-dataset-pile/base"
    splits = ["train"]

    domain_id = PILE_SUBSETS[args.subset]["domain"]
    domain_name = PILE_SUBSETS[args.subset]["name"]
    num_samples = PILE_SUBSETS[args.subset]["raw_samples"]

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.model_max_length = int(1e30)
    columns = {'tokens': 'bytes', 'domain': 'int'}

    for split in splits:
        s3_data = StreamingDataset(remote=s3_remote,
                                   local=local,
                                   split=split,
                                   shuffle=False)
        data = ConcatTokensDataset(dataset=s3_data,
                                   domain=domain_name,
                                   tokenizer=tokenizer,
                                   max_length=2048,
                                   bos_text="",
                                   eos_text="<|endoftext|>",
                                   no_wrap=False)
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader, domain_id, truncate_num_samples=None)

        denominator = _est_progress_denominator(num_samples,
                                                chars_per_sample=6212,
                                                chars_per_token=4,
                                                max_length=2048)

        with MDSWriter(columns=columns,
                       out=os.path.join("/tmp", args.subset, split),
                       compression="zstd") as out:
            for step, sample in enumerate(
                    tqdm(samples, desc=split, total=denominator)):
                out.write(sample)

                if step % 1_000 == 0:
                    wandb.log(({'step': step, 'progress': step / denominator}))

                if step == 10_000:
                    break
