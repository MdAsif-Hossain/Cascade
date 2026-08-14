| Model | Features | Dimensions | Accuracy | Macro-F1 | vs. majority baseline |
|---|---|---:|---:|---:|---:|
| logistic | handcrafted | 23 | 0.389 | 0.227 | -0.456 |
| logistic | embedding | 256 | 0.638 | 0.302 | -0.208 |
| logistic | both | 279 | 0.671 | 0.304 | -0.174 |
| boosting | handcrafted | 23 | 0.832 | 0.329 | -0.013 |
| boosting | embedding | 256 | 0.805 | 0.394 | -0.040 |
| boosting | both | 279 | 0.812 | 0.420 | -0.034 |

Majority-class baseline accuracy: 0.846
