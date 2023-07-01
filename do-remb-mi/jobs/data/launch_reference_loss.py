import argparse
import re
from mcli import RunConfig, create_run

# Probably want a hybrid mode, ie some sort of number of jobs per parallell process to scale to 40K
# TODO: Refactor to use formatting as well
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--model-size", type=str, default="125M")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset-domains", nargs="+", type=int, default=[])
    parser.add_argument("--remote-base", type=str, required=True)
    parser.add_argument("--ref-model-ckpt", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="parallel")
    args = parser.parse_args()

    if not args.subset_domains or len(args.subset_domains) == 0:
        args.subset_domains = [*range(args.num_domains)]

    if args.mode == "parallel":
        for domain_id in range(args.num_domains):
            if domain_id not in args.subset_domains:
                continue
            base_run = RunConfig.from_file(
                f"do-remb-mi/jobs/data-prep/yamls/build_reference_loss.yaml")

            base_run.name = f"ref-loss-domain-{domain_id}"
            base_run.run_name = f"ref-loss-domain-{domain_id}-{args.split}"

            base_run.command = re.sub(r'{SUBSET_DOMAINS}', str(domain_id),
                                      base_run.command)
            base_run.command = re.sub(r'{NUM_DOMAINS}', str(args.num_domains),
                                      base_run.command)
            base_run.command = re.sub(r'{MODEL_SIZE}', str(args.model_size),
                                      base_run.command)
            base_run.command = re.sub(r'{SPLIT}', args.split, base_run.command)
            base_run.command = re.sub(r'{REF_MODEL_CKPT}', args.ref_model_ckpt,
                                      base_run.command)
            base_run.command = re.sub(r'{REMOTE_BASE}', args.remote_base,
                                      base_run.command)
            base_run.command = re.sub(r'{UPLOAD_DIR}', args.upload_dir,
                                      base_run.command)

            launched_run = create_run(base_run)
            print(
                f"Launching reference loss for domain {domain_id} with id: {launched_run.name}"
            )
    elif args.mode == "serial":
        base_run = RunConfig.from_file(
            f"do-remb-mi/jobs/data-prep/yamls/build_reference_loss.yaml")

        base_run.name = "ref-loss-all-domains"
        base_run.run_name = "ref-loss-all-domains"

        base_run.command = re.sub(r'--subset-domains {SUBSET_DOMAINS} ', '',
                                  base_run.command)
        base_run.command = re.sub(r'{NUM_DOMAINS}', str(args.num_domains),
                                  base_run.command)
        base_run.command = re.sub(r'{MODEL_SIZE}', str(args.model_size),
                                  base_run.command)
        base_run.command = re.sub(r'{SPLIT}', args.split, base_run.command)
        base_run.command = re.sub(r'{REF_MODEL_CKPT}', args.ref_model_ckpt,
                                  base_run.command)
        base_run.command = re.sub(r'{REMOTE_BASE}', args.remote_base,
                                  base_run.command)
        base_run.command = re.sub(r'{UPLOAD_DIR}', args.upload_dir,
                                  base_run.command)

        launched_run = create_run(base_run)
        print(
            f"Launching reference loss for all domains with id: {launched_run.name}"
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
