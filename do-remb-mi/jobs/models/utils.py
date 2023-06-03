from typing import List, Optional


def build_domain_streams(num_domains: int, proportions: Optional[List[float]]):
    train_streams = {
        f"domain-{domain_id}": {
            "local":
                f"/tmp/dataset/domain-{domain_id}/",
            "remote":
                f"oci://mosaicml-internal-doremi/pile/pre-concat/gpt-neox-20b-seqlen-2048/data-sources/domain-{domain_id}",
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
