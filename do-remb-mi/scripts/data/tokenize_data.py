import os
import warnings
import platform
from argparse import ArgumentParser 
from typing import Dict, Iterable, Optional

import torch
import wandb
import numpy as np
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils import get_sample_with_uid

torch.multiprocessing.set_sharing_strategy('file_system')

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
            yield get_sample_with_uid({k: v[idx] for k, v in batch.items()})


class TokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes, 'uid': int}

    """

    def __init__(
        self,
        dataset: StreamingDataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'Your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'Your EOS text is not tokenizing to one token\
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
        for sample in self.dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            iids = self.bos_tokens + iids + self.eos_tokens
            yield {
                "tokens": np.asarray(iids).tobytes(),
                "uid": sample["uid"]
            }


if __name__ == "__main__":

    parser = ArgumentParser()
    # Data args
    parser.add_argument("--download-remote", type=str, required=True)
    parser.add_argument("--local", type=str, default="/tmp/tokenize")
    parser.add_argument("--splits",
                        type=str,
                        nargs="+",
                        default=["train", "val", "test"])

    # Tokenizer args
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--tokenizer",
                        type=str,
                        default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--eos-text", type=str, default="<|endoftext|>")
    parser.add_argument("--bos-text", type=str, default=None)

    # Upload args
    parser.add_argument("--upload-remote", type=str, required=True)

    # Misc args
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    if args.bos_text is None:
        args.bos_text = ''
    if args.eos_text is None:
        args.eos_text = ''

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None

        wandb.init(name=args.wandb_name,
                   project="doremi-preprocess",
                   entity="mosaic-ml")

    for split in args.splits:
        print(f"Converting split {split}")
        raw_data = StreamingDataset(remote=args.download_remote,
                                   local=os.path.join(args.local, "downloaded"),
                                   split=split,
                                   shuffle=False)

        tokenizer = tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.model_max_length = int(1e30)

        data = TokensDataset(
            raw_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            bos_text=args.bos_text,
            eos_text=args.eos_text,
        )
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {'tokens': 'bytes', 'uid': 'int'}
        denominator = raw_data.size

        with MDSWriter(columns=columns,
                       out=os.path.join(args.upload_remote, split),
                       compression="zstd") as out:
            for step, sample in enumerate(
                    tqdm(samples, desc=split, total=denominator, leave=True)):
                out.write(sample)

                if use_wandb and step % 1_000 == 0:
                    wandb.log(({'step': step, 'progress': step / denominator}))