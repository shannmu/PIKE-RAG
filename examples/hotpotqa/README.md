# HotpotQA Experimental Results

## Experiments with 500 random sample

| **Method**    | **Setting** | **#Tests** | **#Rounds** | **EM** | **F1** | **Metrics Details** |
|---------------|-------------|------------|-------------|--------|--------|---------------------|
| Direct        |             |    500     |      3      |  44.6  |  53.7  | EM: [44.8, 44.4, 44.6], F1: [53.9, 53.2, 53.9] |
| Q -> Chunk    |             |    500     |      3      |  70.4  |  77.6  | EM: [70.6, 70.2, 70.4], F1: [77.6, 77.5, 77.8] |
| Decomposition | w/ tag v1   |    500     |      3      |  73.6  |  80.7  | EM: [74.2, 73.8, 72.8], F1: [80.8, 81.1, 80.2] |
| Decomposition | w/ tag v2, len limit 640 | 500  |  1   |  69.0  |  76.3  |                     |
| Decomposition | w/ tag v2   |    500     |      3      |  73.4  |  80.7  | EM: [73.6, 74.6, 72.0]; F1: [80.7, 81.8, 79.7] |
| Self-Ask    | w/o retrieval |    500     |      1      |  42.6  |  60.9  |                     |
| Self-Ask    |  w/ retrieval |    500     |      1      |  59.0  |  71.0  |                     |

## Experiments with 500 random sample after dataset protocol applied

|               Method                |    EM    |    F1    |
|:-----------------------------------:|:--------:|:--------:|
|            Zero-Shot CoT            |   32.6   |   43.9   |
|          w/ Chunk Retrieval         |   55.0   |   70.4   |
|        Self-Ask w/ Retrieval        |   45.1   |   62.2   |
| Decompose w/ Hierarchical Retrieval |   59.3   |   74.0   |
