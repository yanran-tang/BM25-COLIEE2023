# BM25-COLIEE2023
This method does French removal preprocessing and employs the ngram BM25 model to select top-k cases from candidate pool. This method achieves competitive F1 score as top submissions of COLIEE 2023 Task 1, `0.299 vs 0.3001`.

# Usage
Download dataset from [COLIEE 2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/).

For training, put the folder `task1_train_files_2023` and label file `task1_train_labels_2023.json` in root `./`, and `$D='train'`.

For testing, put the folder `task1_test_files_2023` and label file `task1_test_labels_2023.json` in root `./`, and `$D='test'`.

Run `python3 bm25.py --ngram_1 $A --ngram_2 $B --topk $C --mode $D`.

# Performance
When `$A`, `$B` are chosen from `1-6` and `$C=5`, the performances of F1 score of training and testing are

Training:
$A \ $B | 1 | 2 | 3 | 4 | 5 | 6 |
--------- | --------|--------| --------| --------| --------| --------|
1 | 0.110 | 0.147 | 0.161 | 0.167 | 0.169 | 0.169 |
2 |       | 0.155 | 0.165 | 0.167 | 0.169 | 0.167 |
3 |       |       | 0.169 | 0.170 | 0.169 | 0.166 |
4 |       |       |       | 0.167 | 0.166 | 0.161 |
5 |       |       |       |       | 0.160 | 0.159 |
6 |       |       |       |       |       | 0.156 |

Testing:
$A \ $B | 1 | 2 | 3 | 4 | 5 | 6 |
--------- | --------|--------| --------| --------| --------| --------|
1 | 0.219 | 0.273 | 0.279 | 0.289 | 0.296 | 0.296 |
2 |       | 0.280 | 0.294 | 0.294 | 0.298 | 0.297 |
3 |       |       | 0.293 | 0.299 | 0.294 | 0.297 |
4 |       |       |       | 0.297 | 0.291 | 0.288 |
5 |       |       |       |       | 0.287 | 0.284 |
6 |       |       |       |       |       | 0.271 |

Top-3 submitted [results](https://sites.ualberta.ca/~rabelo/COLIEE2023/task1_results.html) of COLIEE 2023 Task 1: `0.3001`, `0.2907`, `0.2874`.

# Credit
This repo is based on [nigam](https://github.com/ShubhamKumarNigam/COLIEE-22) repo.