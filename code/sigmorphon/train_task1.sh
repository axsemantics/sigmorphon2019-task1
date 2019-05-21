#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Give language/pair as parameter."
    echo "i.e. bash sigmorphon/train_task1.sh english--north-frisian transfer"
    exit 1
fi

year="2019"
language=$1
meta=$2

if [ -z "${meta}" ]; then
    meta="default"
fi

mkdir -p experiments.${year}
jsonfile=experiments.${year}/$(python sigmorphon/gen_config.py -l ${language} -m ${meta})

directory=output.${year}/$(basename ${jsonfile} .json)
if [ ! -d "$directory" ]; then
    allennlp train ${jsonfile} -s ${directory} --include-package library
fi
