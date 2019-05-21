#!/bin/bash

meta=$1

ls -1 ../conll2019/task1/ | grep -e "\w--\w" | while IFS= read -r line
do
    bash sigmorphon/train_task1.sh ${line} ${meta}
done
