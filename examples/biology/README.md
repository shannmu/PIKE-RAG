# Biology Q&A

## Text Book

The used biology book is hosted in GitHub [biology-book](https://github.com/philschatz/biology-book).

The Table of Contents is in [./SUMMARY.md](./SUMMARY.md)

## Test Dataset

The used test dataset is [MMLU/college_biology](https://huggingface.co/datasets/cais/mmlu).

## Pipeline and Script Description

Assume that you are in the directory `examples/`:

```sh
# Using the general chunking script to process original documents.
python chunking.py biology/configs/chunking.yml

# Using the general qa script to do Q&A testing on the given testing suite.
python qa.py biology/configs/qa.yml
```

## Experimental Results

- 3 rounds for each

### GPT-4

| **Reference** | **Prompt** | **Retrieval Query** | **GPT-4** | **3 Rounds**        | **Notes** |
|:-------------:|------------|---------------------|:---------:|---------------------|-----------|
|               |            |                     |   94.68   | 93.06, 95.14, 95.83 |           |
|     chunks    |            |  Question (single)  |   93.98   | 93.75, 93.75, 94.44 |           |
|     chunks    |            |   Q + Opts (single) |   94.44   | 93.06, 93.75, 96.53 |           |
|     chunks    |            | Q, Each Opts (multi)|   91.90   | 87.50, 93.75, 94.44 |           |
|     chunks    |            | Q + Each Opt (multi)|   92.82   | 92.36, 93.06, 93.06 |           |
|     chunks    |            | Q + Each Opt (multi)|   94.68   | 93.75, 94.44, 95.83 | 32, .6    |
|     chunks    |  w/ review |   Q + Opts (single) | **96.99** | 96.53, 97.22, 97.22 |           |
|     chunks    |  w/ review | Q + Each Opt (multi)|   91.90   | 90.28, 92.36, 93.06 |           |

### GPT-35-turbo

| **Reference** | **Prompt** | **Retrieval Query** |**35-turbo**| **3 Rounds**       | **Notes** |
|:-------------:|------------|---------------------|:---------:|---------------------|-----------|
|               |            |                     |   75.93   | 73.61, 75.00, 79.17 |           |
|     chunks    |            |  Question (single)  |   76.62   | 75.00, 77.08, 77.78 |           |
|     chunks    |            |  Question (single)  |   70.83   | 69.44, 70.83, 72.22 | 16, .2    |
|     chunks    |            |   Q + Opts (single) | **80.32** | 77.78, 80.56, 82.64 |           |
|     chunks    |            | Q, Each Opts (multi)|   77.78   | 75.69, 77.08, 80.56 |           |
|     chunks    |            | Q + Each Opt (multi)|   77.31   | 77.08, 77.08, 77.78 |           |
|     chunks    |  w/ review |   Q + Opts (single) |   77.31   | 77.08, 77.08, 77.78 |           |
|     chunks    |  w/ review | Q + Each Opt (multi)| **79.17** | 76.39, 77.08, 84.03 |           |
