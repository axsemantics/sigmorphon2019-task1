## Code of "AX Semantics' Submission to the SIGMORPHON 2019 Shared Task"

Paper: https://www.aclweb.org/anthology/papers/W/W19/W19-4201/

### Introduction

We use AllenNLP - https://github.com/allenai/allennlp

The given configurations require CUDA.
To disable set ``cuda_device: -1`` in configuration files.

### Checkout data

```
git clone https://github.com/sigmorphon/2019 conll2019
```

### folder structure in code

| path           | description                              |
|----------------|------------------------------------------|
| library        | python code that extends allennlp        |
| configurations | experiments based on this configurations |
| sigmorphon     | sigmorphon specific files and scripts    |


### run code

```
cd code
pip install -r requirements.txt

# system1

bash sigmorphon/all_train_2019task1.sh

# system2

bash sigmorphon/all_train_2019task1.sh transfer

```

### predict for all trained models

```
bash sigmorphon/all_predict_2019task1.sh
```
