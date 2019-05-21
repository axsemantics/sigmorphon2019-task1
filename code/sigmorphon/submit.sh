#!/bin/bash

base=submission/AX-01-1

for eval_folder in `ls -1d eval.2019/system1*`; do
    language=$(echo ${eval_folder} | rev | cut -d'_' -f1 | rev)
    low_language=$(echo $language|sed -e 's/--/:/g'|cut -d: -f2)

    echo ${language}

    folder="${base}/${language}"
    mkdir -p ${folder}
    cp ${eval_folder}/${language}-test-covered ${folder}/${low_language}-test.output
done

base=submission/AX-02-1

for eval_folder in `ls -1d eval.2019/system2_*`; do
    language=$(echo ${eval_folder} | rev | cut -d'_' -f1 | rev)
    low_language=$(echo $language|sed -e 's/--/:/g'|cut -d: -f2)

    echo ${language}

    folder="${base}/${language}"
    mkdir -p ${folder}
    cp ${eval_folder}/${language}-test-covered ${folder}/${low_language}-test.output
done

(
    cd submission
    rm -f AX-01-1.tgz
    tar czf AX-01-1.tgz AX-01-1
    rm -f AX-02-1.tgz
    tar czf AX-02-1.tgz AX-02-1
)
