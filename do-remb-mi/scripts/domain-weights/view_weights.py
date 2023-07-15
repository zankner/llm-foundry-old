import argparse
import os
import re
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
BASELINE_PROPORTIONS = [
    0.16, 0.127, 0.114, 0.09, 0.102, 0.098, 0.055, 0.056, 0.029, 0.024, 0.021,
    0.017, 0.037, 0.021, 0.012, 0.007, 0.009, 0.006, 0.008, 0.004, 0.002, 0.001
]
CLUSTERS_22_PROPORTIONS = [
    0.0384240625, 0.03094484375, 0.019497421875, 0.021077734375, 0.09471203125,
    0.142099375, 0.016896328125, 0.035976796875, 0.0302825, 0.0598478125,
    0.119653671875, 0.00718921875, 0.02127328125, 0.009433046875, 0.0166284375,
    0.07820453125, 0.054911640625, 0.023267421875, 0.0343740625, 0.032515625,
    0.08241453125, 0.030375625
]
REPLICATE_PROPORTIONS = [
    0.20163412392139435, 0.10023175925016403, 0.13405060768127441,
    0.1319042444229126, 0.07527627050876617, 0.09094192832708359,
    0.05040840804576874, 0.04523293673992157, 0.024078955873847008,
    0.03058076836168766, 0.019306106492877007, 0.0025820545852184296,
    0.0365639366209507, 0.002238696441054344, 0.00817348062992096,
    0.0037490464746952057, 0.016184765845537186, 0.004014396108686924,
    0.008683200925588608, 0.007775607518851757, 0.004500322509557009,
    0.0018897337140515447
]
WEIGHTS_BASE = "oci://mosaicml-internal-checkpoints/zack/DoReMi/proxy"


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


def get_compare_weights(compare_run_name):
    compare_weights_path = os.path.join(WEIGHTS_BASE, compare_run_name,
                                        "domain-weights", "final",
                                        "average_domain_weights.npy")
    compare_weights = load_weights(compare_weights_path)
    return compare_weights


def ema(weights, decay=0.99):
    ema_weights = []
    for weight in weights:
        if len(ema_weights) == 0:
            ema_weights.append(weight)
        else:
            ema_weights.append(decay * ema_weights[-1] + (1 - decay) * weight)
    return ema_weights


def main(args):
    weights_dir = os.path.join(WEIGHTS_BASE, args.run_name, "domain-weights")
    weights = get_all_weights(weights_dir,
                              num_domains=args.num_domains,
                              max_steps=args.max_steps,
                              start_step=args.start_step,
                              log_freq=args.log_freq,
                              average=not args.ema)

    if args.compare_run_name == "pile":
        compare_weights = BASELINE_PROPORTIONS
    elif args.compare_run_name == "22-clusters":
        compare_weights = CLUSTERS_22_PROPORTIONS
    elif args.compare_run_name == "replicate":
        compare_weights = REPLICATE_PROPORTIONS
    elif args.compare_run_name == "prev-iter":
        cur_iter = int(re.search(r"iter-(\d+)", args.run_name).group(1))
        compare_run_name = re.sub(r"iter-(\d+)",
                                  lambda match: "iter-" + str(cur_iter - 1),
                                  args.run_name)
        compare_weights = get_compare_weights(compare_run_name)
    else:
        compare_weights = get_compare_weights(args.compare_run_name)

    if args.ema:
        weights = {k: ema(v, args.ema) for k, v in weights.items()}
        sums = [
            sum([weights[weight][i]
                 for weight in weights])
            for i in range(len(weights["domain-0"]))
        ]
        weights = {
            k: [v[i] / sums[i] for i in range(len(v))]
            for k, v in weights.items()
        }

    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"

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

        og_proportion = compare_weights[int(domain_name.split('-')[-1])]
        delta = weight_trajectory[-1] - og_proportion
        print(
            f"{display_domain_name}: {weight_trajectory[-1]} ---- Delta: {green if delta >=0 else red}{delta:.3f}  ({(delta / og_proportion)*100:.2f}%){reset}"
        )

    # Create the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Adjust the layout to accommodate the legend
    plt.subplots_adjust(right=0.7)  # Increase the right margin

    plt.title(args.title)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--compare-run-name", type=str, default="baseline")
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--title", type=str, default="Domain Weights")
    parser.add_argument("--data-source", action="store_true")
    parser.add_argument("--ema", type=float, default=None)
    args = parser.parse_args()

    main(args)
