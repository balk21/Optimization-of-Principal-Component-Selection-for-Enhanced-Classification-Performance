# MNIST: k picked by each rule and test accuracy (%)

standard split, 60 000 training and 10 000 test images.

| rule | k | 1-nearest neighbour | logistic regression |
|---|---|---|---|
| derivative rule of the report | 691 | 96.91 | 92.63 |
| knee of the error curve | 129 | 97.04 | 92.42 |
| 95 % explained variance | 154 | 96.94 | 92.29 |
| all components | 784 | 96.91 | 92.63 |
| best k on the test set (reference, not a rule) | | 97.35 (k = 40) | 92.64 (k = 650) |

