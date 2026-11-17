# What the report's derivative rule depends on

In-sample error curves, as in the report. Each cell is the k the rule selects.

## AT&T (random subsets, 5 per size; mean and range)

| images fitted | rank of the data | step 1 | step 5 | step 10 | k / images (step 5) |
|---|---|---|---|---|---|
| 100 | 99 | 100 | 65 (56–71) | 61 | 0.65 |
| 200 | 199 | 200 | 134 (126–141) | 133 (131–141) | 0.67 |
| 300 | 299 | 300 | 202 (186–206) | 207 (191–211) | 0.67 |
| 400 | 399 | 400 | 311 | 281 | 0.78 |

All 400 images, step 5, the notebooks' procedure (sklearn's default solver) run 5 times: k = 286, 276, 271, 296, 276. The report's 271 is one such run.

## MNIST (the first n images, as in the report)

| images fitted | rank of the data | step 1 | step 10 |
|---|---|---|---|
| 1000 | 590 | 589 | 601 |
| 2000 | 625 | 624 | 631 |
| 5000 | 653 | 653 | 661 |
| 10000 | 664 | 664 | 671 |
| 70000 | 713 | 680 | 691 |
