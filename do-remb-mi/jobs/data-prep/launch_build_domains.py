import argparse
import re
from mcli import RunConfig, create_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-domains", type=int, required=True)
    parser.add_argument("--remote", type=str, required=True)
    parser.add_argument("--domain-type", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--wandb-name", type=str, required=True)
    args = parser.parse_args()

    base_run = RunConfig.from_file(
        f"do-remb-mi/jobs/data-prep/yamls/build_domains.yaml")

    base_run.command = re.sub(r'{DOWNLOAD_DIR}', args.remote, base_run.command)
    base_run.command = re.sub(r'{DOMAIN_TYPE}', args.domain_type,
                              base_run.command)
    base_run.command = re.sub(r'{NUM_DOMAINS}', str(args.num_domains),
                              base_run.command)
    base_run.command = re.sub(r'{WANDB_NAME}', args.wandb_name,
                              base_run.command)
    base_run.command = re.sub(r'{UPLOAD_DIR_PREFIX}', args.upload_dir,
                              base_run.command)

    launched_run = create_run(base_run)
    print(f"Launching build domains with id: {launched_run.name}")
