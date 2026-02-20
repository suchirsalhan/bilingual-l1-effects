#!/bin/bash

for lang in es fr de pl tr ar zh; do
    echo "Submitting job for language: $lang"
    sbatch launch_slurm.wilkes3 $lang
done
