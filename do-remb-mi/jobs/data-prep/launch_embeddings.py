do-remb-mi/jobs/data-prep/launch_embeddings.py
import re
from mcli import RunConfig, create_run

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
subsets = list(PILE_SUBSETS.values())
oom_runs = [
    "books3", "gutenberg-pg-19", "ubuntu-irc", "books-corpus-2", "euro-parl"
]
completed_runs = [
    "dm-mathematics", "enron-emails", "free-law", "hacker-news", "nih-exporter",
    "open-subtitles", "phil-papers", "pile-cc", "pubmed-central",
    "uspto-backgrounds", "youtube-subtitles"
]

for subset in oom_runs:

    base_run = RunConfig.from_file(
        f"do-remb-mi/jobs/data-prep/yamls/build_embeddings.yaml")

    base_run.name = f"embeddings-{subset}"
    base_run.run_name = f"embeddings-{subset}"

    base_run.command = re.sub(r'{SUBSET}', subset, base_run.command)

    launched_run = create_run(base_run)
    print(f"Launching embeddings for {subset} with id: {launched_run.name}")