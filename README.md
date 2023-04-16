# BM25-COLIEE2023
This method does French removal preprocessing and employs the ngram BM25 model to select top-k cases from candidate pool. This method achieves competitive F1 score as top submissions of COLIEE 2023 Task 1, `0.299 vs 0.3001`.

# Usage
Download dataset from [COLIEE 2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/).
Put the folder `task1_test_files_2023` and label file `task1_test_labels_2023.json` in root `./`.

Run `python3 bm25.py --ngram_1 $A --ngram_2 $B`.

# Performance
When `$A`, `$B` are chosen from `1-6`, the performances of F1 score are

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