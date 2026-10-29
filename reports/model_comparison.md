| model | cv_f1 | precision | recall | f1 | roc_auc | false_alarms | missed_scams | seconds |
|---|---|---|---|---|---|---|---|---|
| Complement Naive Bayes (word TF-IDF) | 0.9022 | 0.8667 | 0.9353 | 0.8997 | 0.9917 | 20 | 9 | 0.6 |
| Linear SVM (word TF-IDF) | 0.9375 | 0.9357 | 0.9424 | 0.9391 | 0.9955 | 9 | 8 | 0.6 |
| SGD / modified Huber (word TF-IDF) | 0.9264 | 0.9097 | 0.9424 | 0.9258 | 0.9892 | 13 | 8 | 0.5 |
| Logistic Regression (word TF-IDF) | 0.9322 | 0.9362 | 0.9496 | 0.9429 | 0.9955 | 9 | 7 | 0.6 |
| Logistic Regression (word+char+hand features) [ours] | 0.9528 | 0.9854 | 0.9712 | 0.9783 | 0.9984 | 2 | 4 | 4.5 |
