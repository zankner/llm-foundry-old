import os
import copy
from typing import List, Optional


def build_ref_name(args, domain_types):
    return f"ref-{args.model_size}-{args.dataset}-{domain_types}-{args.subsample_dist}-{args.num_samples}"


def build_proxy_name(args, domain_types):
    ref_args = copy.deepcopy(args)
    ref_args.subsample_dist = args.ref_subsample_dist
    return f"proxy-ss-{args.step_size}-sm-{args.smoothing}-ws-{args.warmup_steps}-iter-{args.iter}-{build_ref_name(ref_args, domain_types)}"


def build_final_name(args, domain_types):
    if args.domain_source == "doremi":
        proxy_args = copy.deepcopy(args)
        proxy_args.model_size = args.proxy_model_size
        return f"final-{args.model_size}-{build_proxy_name(proxy_args, domain_types)}"
    return f"final-{args.model_size}-{args.domain_source}-{args.dataset}-{args.num_samples}"


def build_data_path(args, mode):
    data_prefix = os.path.join("oci://mosaicml-internal-doremi", args.dataset,
                               mode, args.tokenizer)

    embed = not args.not_embed
    if embed:
        domain_dir = "data-sources"
    else:
        domain_dir = f"{args.num_domains}-clusters"

    subsample_dir = f"baseline-{args.num_samples}-samples"

    remote_base = os.path.join(data_prefix, domain_dir, subsample_dir)
    return remote_base, domain_dir


# domain streams handle different remotes for different app
def build_domain_streams(num_domains: int,
                         remote_base: str,
                         proportions: Optional[List[float]] = None):

    train_streams = {
        f"domain-{domain_id}": {
            "local": f"/tmp/dataset/domain-{domain_id}/",
            "remote": os.path.join(remote_base, f"domain-{domain_id}"),
            "split": "train",
        } for domain_id in range(num_domains)
    }
    if proportions is not None:
        train_streams = {
            domain_id: {
                **stream, "proportion": proportion
            } for (domain_id,
                  stream), proportion in zip(train_streams.items(), proportions)
        }
    return train_streams
