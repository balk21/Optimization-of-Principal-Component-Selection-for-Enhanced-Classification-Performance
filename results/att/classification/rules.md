# AT&T: k picked by each rule and test accuracy (%)

10 random splits, 5 training and 5 test images per person.

| rule | k | 1-nearest neighbour | logistic regression |
|---|---|---|---|
| derivative rule of the report | 137 (126–151) | 93.8 ± 1.0 | 95.6 ± 1.5 |
| knee of the error curve | 34 (32–35) | 93.6 ± 1.2 | 95.5 ± 1.5 |
| 95 % explained variance | 115 (114–116) | 93.8 ± 1.0 | 95.8 ± 1.3 |
| all components | 200 | 93.8 ± 0.8 | 95.7 ± 1.4 |
| best k on the test set (reference, not a rule) | | 94.1 (k = 50) | 96.0 (k = 85) |

k: mean (range) over splits. Accuracy: mean ± standard deviation over splits.
Knee minus derivative rule, paired by split: 1-nearest neighbour -0.1 ± 0.8 points, logistic regression -0.1 ± 0.9 points.
