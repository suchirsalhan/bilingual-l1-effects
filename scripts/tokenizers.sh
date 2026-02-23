#!/bin/bash
#
# Submit one tokenizer job per language + source combo
#

LANGS=("es" "fr" "de" "pl" "tr" "ar" "zh")
SOURCES=("fineweb" "redpajama")

for lang in "${LANGS[@]}"; do
    for src in "${SOURCES[@]}"; do
        echo "Submitting job for language=$lang, source=$src"
        sbatch launch_tokenizer.wilkes3.sh "$lang" "$src"
    done
done
