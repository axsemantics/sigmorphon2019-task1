#!/bin/bash

if [ "$#" -le 1 ]; then
    echo "Give the output folder as first parameter."
    echo "i.e. bash sigmorphon/predict.sh output.2019/simple_seq2seq_russian--portuguese"
    exit 1
fi

output_folder=$1
affix=$2
year="2019"
language=$(echo ${output_folder} | rev | cut -d'_' -f1 | rev)
base="../conll2019/task1/"

if [ -z "${affix}" ]; then
    affix=test-covered
fi

test_fn=${base}/${language}-${affix}

lowreslang=$(echo $language|sed -e 's/--/:/g'|cut -d: -f2)
test_fn=${base}/${language}/${lowreslang}-${affix}

eval_path=eval.${year}/$(basename ${output_folder})
mkdir -p ${eval_path}

eval_fn=${eval_path}/${language}-${affix}.output
eval_stdout=${eval_path}/stdout-${affix}.log
eval_stderr=${eval_path}/stderr-${affix}.log

if [ -e "${eval_fn}" ]; then
    exit 0
fi

echo "predicting"
allennlp predict ${output_folder}/model.tar.gz --use-dataset-reader ${test_fn} --predictor seq2seq --include-package library --output-file ${eval_fn} > ${eval_stdout} 2> ${eval_stderr}
