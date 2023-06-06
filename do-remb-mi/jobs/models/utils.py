import os
from typing import List, Optional

REMOTE_BASE = "oci://mosaicml-internal-doremi/pile/pre-concat/gpt-neox-20b-seqlen-2048"


# domain streams handle different remotes for different app
def build_domain_streams(num_domains: int,
                         remote_dir: str,
                         proportions: Optional[List[float]] = None,
                         ref_dataset: bool = False):
    if ref_dataset:
        remote_base = "oci://mosaicml-internal-doremi/pile/token-ref-loss/gpt-neox-20b-seqlen-2048"
    else:
        remote_base = "oci://mosaicml-internal-doremi/pile/pre-concat/gpt-neox-20b-seqlen-2048"
    train_streams = {
        f"domain-{domain_id}": {
            "local":
                f"/tmp/dataset/domain-{domain_id}/",
            "remote":
                os.path.join(remote_base, remote_dir, f"domain-{domain_id}"),
            "split":
                "train",
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
