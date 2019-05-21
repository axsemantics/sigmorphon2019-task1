#!/bin/bash

year=2019
affix=$1

if [ -z "${affix}" ]; then
    # affix=dev  # for development
    affix=test
fi

base=../conll2019
predict_log="eval.2019/000_predict_${affix}.log"
tmpfile=$(mktemp /tmp/all_predict_2019task1.XXXXXX)

ls -1 output.2019 | while IFS= read -r line
do
    echo ${line}
    output_folder=output.2019/$line

    if [ ! -e "${output_folder}/model.tar.gz" ]; then
        echo "${output_folder}/model.tar.gz doesn't exist!"
        continue
    fi

    language=$(echo ${output_folder} | rev | cut -d'_' -f1 | rev)
    eval_path="eval.${year}/$(basename ${output_folder})"

    bash sigmorphon/predict.sh ${output_folder} ${affix}

    # convert result to sigmorphon format
    python sigmorphon/predict_to_sigmorphon.py -l ${language} -a ${affix} -ep ${eval_path}

    # run eval script from organizers
    low=$(echo $line|sed -e 's/--/:/g'|cut -d: -f2)
    result=$(python ${base}/evaluation/evaluate_2019_task1.py -r ${base}/task1/${language}/${low}-${affix} -o ${eval_path}/${language}-${affix})
    acc=$(echo $result | cut -d' ' -f2 | cut -d':' -f2)
    dist=$(echo $result | cut -d' ' -f4 | cut -d':' -f2)

    if [ ! -e ${predict_log} ]; then
        touch ${predict_log}
    fi

    # remove line(s) from predict log and add new one
    grep -v "${line}" ${predict_log} > ${tmpfile}
    echo "| ${line} | ${language} | ${acc} | ${dist} |">> ${tmpfile}
    cp ${tmpfile} ${predict_log}
done

rm "${tmpfile}"
