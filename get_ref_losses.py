from streaming import StreamingDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]

losses = []
for i in tqdm(range(22)):
    ds = StreamingDataset(
        remote=
        f"oci://mosaicml-internal-doremi/pile/token-ref-loss/gpt-neox-20b-seqlen-2048/data-sources/domain-{i}",
        split="train")
    n_samples = ds.index.total_samples
    ds = DataLoader(ds, batch_size=2048, num_workers=64, prefetch_factor=64)
    loss = 0
    for batch in tqdm(ds):
        for sample in batch["ref_losses"]:
            loss += np.frombuffer(sample['ref_losses'],
                                  dtype=np.float16)[:2048].copy().sum()
    loss = loss / (n_samples * 2048)
    losses.append(loss)

print(losses)

for i, loss in enumerate(losses):
    print(f"Domain {PILE_DATA_SOURCES[i]}: {loss}")