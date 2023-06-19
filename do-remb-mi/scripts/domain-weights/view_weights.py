import argparse
import os
import tempfile
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from composer.utils import get_file

PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]
WEIGHTS_BASE = "oci://mosaicml-internal-doremi/pile/proxy-weights"


def load_weights(weight_file: str):
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_file(weight_file, tmp_file.name, overwrite=True)
        weights = np.load(tmp_file.name)
    return weights


def get_all_weights(weights_dir: str,
                    num_domains: int,
                    max_steps: int,
                    start_step: int,
                    log_freq: int,
                    average: bool = True):
    suffix = "average_domain_weights.npy" if average else "domain_weights.npy"
    weights = {f"domain-{domain_idx}": [] for domain_idx in range(num_domains)}
    for batch_idx in range(start_step, max_steps + 1, log_freq):
        weight_path = os.path.join(weights_dir, f"ba-{batch_idx}", suffix)
        batch_weights = load_weights(weight_path)
        for domain_idx, domain_weight in enumerate(batch_weights):
            weights[f"domain-{domain_idx}"].append(domain_weight)
    return weights


def ema(weights, decay=0.99):
    ema_weights = []
    for weight in weights:
        if len(ema_weights) == 0:
            ema_weights.append(weight)
        else:
            ema_weights.append(decay * ema_weights[-1] + (1 - decay) * weight)
    return ema_weights


def main(args):
    weights_dir = os.path.join(WEIGHTS_BASE, args.run_name)
    weights = get_all_weights(weights_dir,
                              num_domains=args.num_domains,
                              max_steps=args.max_steps,
                              start_step=args.start_step,
                              log_freq=args.log_freq,
                              average=not args.no_average)
    if args.ema:
        weights = {k: ema(v) for k, v in weights.items()}
        sums = [
            sum([weights[weight][i]
                 for weight in weights])
            for i in range(len(weights["domain-0"]))
        ]
        weights = {
            k: [v[i] / sums[i] for i in range(len(v))]
            for k, v in weights.items()
        }

    # Sort the weights
    weights = sorted(weights.items(), key=lambda x: x[1][-1], reverse=True)

    # Plot the weights
    steps = list(range(args.start_step, args.max_steps + 1, args.log_freq))
    for domain_name, weight_trajectory in weights:
        # domain_name = f"domain-{domain_idx}"
        display_domain_name = domain_name
        if args.data_source:
            display_domain_name = PILE_DATA_SOURCES[int(
                domain_name.split("-")[-1])]
        plt.plot(steps, weight_trajectory, label=display_domain_name)
        print(f"{display_domain_name}: {weight_trajectory[-1]}")

    # Create the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the layout to accommodate the legend
    plt.subplots_adjust(right=0.7)  # Increase the right margin

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--data-source", action="store_true")
    parser.add_argument("--no-average", action="store_true")
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    main(args)
