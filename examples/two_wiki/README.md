# 2WikiMultihop Experimental Results

## Experiments with 500 random sample

| **Method**    | **Setting** | **#Tests** | **#Rounds** | **EM** | **F1** | **Metrics Details** |
|---------------|-------------|------------|-------------|--------|--------|---------------------|
| Direct        |             |    500     |      3      |  37.73 |  42.4  | EM: [36.6, 38.8, 37.8], F1: [41.6, 43.2, 42.5] |
| Q -> Chunk    |             |    500     |      3      |  51.73 |  55.3  | EM：[51.8, 51.6, 51.8], F1: [55, 55.4, 55.5] |
| Decomposition | w/ tag v1   |    500     |      3      |  76.53 |  80.8  | EM: [77.2, 76.2, 76.2], F1: [81.5, 80.6, 80.2] |
| Self-Ask     | w/ retrieval |    500     |      1      |  54.8  |  71.7  |                     |

## Experiments with 500 random sample after dataset protocol applied

|               Method                |    EM    |    F1    |
|:-----------------------------------:|:--------:|:--------:|
|            Zero-Shot CoT            |   35.7   |   41.4   |
|          w/ Chunk Retrieval         |   51.5   |   59.5   |
|        Self-Ask w/ Retrieval        |   48.8   |   65.3   |
| Decompose w/ Hierarchical Retrieval |   65.9   |   74.3   |
