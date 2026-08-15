| Model | Features | Dimensions | Accuracy | Macro-F1 | vs. majority baseline |
|---|---|---:|---:|---:|---:|
| logistic | handcrafted | 22 | 0.396 | 0.230 | -0.450 |
| logistic | embedding | 256 | 0.638 | 0.302 | -0.208 |
| logistic | both | 278 | 0.671 | 0.304 | -0.174 |
| boosting | handcrafted | 22 | 0.846 | 0.334 | +0.000 |
| boosting | embedding | 256 | 0.805 | 0.394 | -0.040 |
| boosting | both | 278 | 0.819 | 0.423 | -0.027 |

Majority-class baseline accuracy: 0.846
