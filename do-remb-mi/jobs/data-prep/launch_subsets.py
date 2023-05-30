from mcli import RunConfig, create_run

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
subsets = list(PILE_SUBSETS.keys())

for subset in subsets:
    base_run = RunConfig.from_file(
        f"do-remb-mi/jobs/data-prep/yamls/build_subset_pile.yaml")

    base_run.run_name = f"pile-{subset}"

    run_command = f"python scripts/data_prep/build_subsets.py --subset {subset}"
    upload_command = f"oci os object bulk-upload -bn mosaicml-internal-doremi --src-dir /tmp/{subset} --prefix data-sources/{subset}/pre-concat/gpt-neox-20b-seq-len-2048"

    base_run.command += run_command + "\n" + upload_command

    if subset in ["pile-cc", "pubmed-central", "books3"]:
        base_run.scheduling["priority"] = "medium"
    else:
        base_run.scheduling["priority"] = "low"

    print(base_run)
    print()
    print()