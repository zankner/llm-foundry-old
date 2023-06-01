import os
import platform
from argparse import ArgumentParser
from typing import Dict, Optional, Iterable

from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import wandb

PILE_SUBSETS = {
    "Pile-CC": "pile-cc",
    "PubMed Central": "pubmed-central",
    "Books3": "books3",
    "OpenWebText2": "open-web-text-2",
    "ArXiv": "arxiv",
    "Github": "github",
    "FreeLaw": "free-law",
    "StackExchange": "stack-exchange",
    "USPTO Backgrounds": "uspto-backgrounds",
    "PubMed Abstracts": "pubmed-abstracts",
    "Gutenberg (PG-19)": "gutenberg-pg-19",
    "OpenSubtitles": "open-subtitles",
    "Wikipedia (en)": "wikipedia-en",
    "DM Mathematics": "dm-mathematics",
    "Ubuntu IRC": "ubuntu-irc",
    "BookCorpus2": "books-corpus-2",
    "EuroParl": "euro-parl",
    "HackerNews": "hacker-news",
    "YoutubeSubtitles": "youtube-subtitles",
    "PhilPapers": "phil-papers",
    "NIH ExPorter": "nih-exporter",
    "Enron Emails": "enron-emails"
}


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
            pile_set_name = batch['pile_set_name'][idx]
            n_samples += 1
            yield {
                **{
                    k: v[idx] for k, v in batch.items() if k != "uid"
                }, 'uid': batch['uid'][idx].item()
            }, pile_set_name


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes, 'pile_set_name': str, 'uid': int}
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.dataset:
            yield {
                'text': sample['text'],
                'pile_set_name': sample['pile_set_name'],
                'uid': sample['uid']
            }


# TODO: REFAC TO TAKE IN THE EMBEDDINGS FOR CLUSTERS
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--remote", type=str, required=True)
    parser.add_argument("--local", type=str, default="/tmp/subset-local")
    parser.add_argument("--splits",
                        type=str,
                        nargs="+",
                        default=["train", "val", "test"])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        assert args.wandb_name is not None

        wandb.init(name=args.wandb_name,
                   project="doremi-preprocess",
                   entity="mosaic-ml")

    for split in args.splits:
        print(f"Converting split {split}")
        streaming_data = StreamingDataset(remote=args.remote,
                                          local=args.local,
                                          split=split,
                                          shuffle=False)

        data = NoConcatDataset(streaming_data)
        loader = build_dataloader(data, batch_size=512)
        samples = generate_samples(loader, truncate_num_samples=None)

        columns = {'text': 'str', 'pile_set_name': 'str', 'uid': 'int'}
        denominator = 210607728

        writers = {
            pile_set_name:
                MDSWriter(columns=columns,
                          out=os.path.join("/tmp", "subsets", subset, split),
                          compression="zstd")
            for pile_set_name, subset in PILE_SUBSETS.items()
        }
        for step, (sample, pile_set_name) in enumerate(
                tqdm(samples, desc=split, total=denominator, leave=True)):

            writers[pile_set_name].write(sample)

            if use_wandb and step % 1_000 == 0:
                wandb.log(({'step': step, 'progress': step / denominator}))

        for writer in writers.values():
            writer.finish()
