# Model comparison

734 training and 184 hold-out patients (80/20 split stratified on site x class). CV = 5 stratified folds of the training split; test = the hold-out, never used for selection. The champion has the best mean CV ROC-AUC. Threshold 0.5.

| Family | CV ROC-AUC | CV Accuracy | CV Precision | CV Recall | CV F1 | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_majority | 0.500 ± 0.000 | 0.553 | 0.553 | 1.000 | 0.712 | 0.500 | 0.554 | 0.554 | 1.000 | 0.713 |
| **logistic_regression** (champion) | 0.907 ± 0.029 | 0.834 | 0.853 | 0.845 | 0.849 | 0.906 | 0.842 | 0.829 | 0.902 | 0.864 |
| random_forest | 0.897 ± 0.023 | 0.823 | 0.829 | 0.857 | 0.843 | 0.896 | 0.853 | 0.844 | 0.902 | 0.872 |
| hist_gradient_boosting | 0.895 ± 0.026 | 0.811 | 0.815 | 0.852 | 0.833 | 0.900 | 0.853 | 0.844 | 0.902 | 0.872 |

## Leave-one-site-out (logistic_regression)

Champion hyper-parameters, trained on the other three sites, scored on the fourth.

| Held-out site | Patients | Disease | ROC-AUC | Accuracy | Recall |
| --- | --- | --- | --- | --- | --- |
| Cleveland | 303 | 46% | 0.876 | 0.762 | 0.878 |
| Hungary | 293 | 36% | 0.902 | 0.795 | 0.906 |
| Switzerland | 123 | 93% | 0.749 | 0.772 | 0.791 |
| VA Long Beach | 199 | 74% | 0.730 | 0.759 | 0.831 |
