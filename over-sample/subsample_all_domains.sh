#!/bin/bash

domains=('c4' 'markdown' 'mc4' 'redpajama' 'redpajama-arxiv' 'redpajama-books' 'redpajama-stackexchange' 'redpajama-wiki' 's2' 'stack')

for domain_name in "${domains[@]}"
do
  python subsample_dataset.py --domain ${domain_name} --available-num-tokens 13B --seed 17
done
