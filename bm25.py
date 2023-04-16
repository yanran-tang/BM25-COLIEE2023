import os
import codecs
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

import langdetect
from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory
DetectorFactory.seed = 0

from bm25_model import BM25, micro_prec

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## Training config of BM25
parser.add_argument("--ngram_1", type=int, default=4,
                    help="ngram")
parser.add_argument("--ngram_2", type=int, default=4,
                    help="ngram")
parser.add_argument("--topk", type=int, default=5,
                    help="select top-k cases as final prediction")
args = parser.parse_args()

## Data labels
with open('./task1_test_labels_2023.json', 'r') as f:
    noticed_case_list = json.load(f)
    f.close()

## Files paths
RDIR = './task1_test_files_2023'
WDIR = './preprocessed_files'

## Data preprocessing starts
files = os.listdir(RDIR)
for pfile in tqdm(files[:]):

    # check if preproceed file exists
    if os.path.exists(os.path.join(WDIR, pfile)):
        continue
    else:
        para = ''
        para_list = []
        with open(os.path.join(RDIR, pfile), 'r') as f:
            filetx = f.read()
            f.close()
        num = 1
        text = filetx.split()
        for i in text:
            if i == str([num]):
                try:
                    if str(detect_langs(para)[0])[:6]=='fr:0.9':
                        # print(i, para)
                        pass
                        # print(detect_langs(j.lower()))
                    else:
                        para_list.append(para)
                except langdetect.lang_detect_exception.LangDetectException:
                    pass
                num += 1
                para = ''
            elif i == text[-1]:
                para += i
                try:
                    if str(detect_langs(para)[0])[:6]=='fr:0.9':
                        # print(i, para)
                        pass
                        # print(detect_langs(j.lower()))
                    else:
                        # print(i, para)
                        para_list.append(para)
                except langdetect.lang_detect_exception.LangDetectException:
                    pass            
            else:
                para += i+' '      
        
        with open(os.path.join(WDIR, pfile), 'w') as file:
            json.dump(para_list, file, ensure_ascii=False)
            file.close()
## Data preprocessing ends

## dataset path
my_suffixes = (".txt")
citation_file_paths = []
# r=root, d=directories, f = files
for r, d, f in os.walk(WDIR):
#     print(r,len(r))
    for file in f:
#         print(file)
        if file.endswith(my_suffixes):
            citation_file_paths.append(os.path.join(r, file))
name_dict = {}
corpus =[]
citation_names = []
for file in sorted(citation_file_paths):
    f = codecs.open(file, "r", "utf-8", errors='ignore') 
    text = f.read()
    corpus.append(text)
    citation_names.append(os.path.basename(file))
    name_dict[text] = os.path.basename(file)
# len(corpus)

bm25 = BM25(ngram_range=(args.ngram_1, args.ngram_2))
bm25.fit(corpus)

query_file_paths = []
# r=root, d=directories, f = files
for r, d, f in os.walk(WDIR):
    #     print(r,len(r))
    for file in f:
        if file.split('/')[-1] in noticed_case_list.keys():
            #         print(file)
            if file.endswith(my_suffixes):
                query_file_paths.append(os.path.join(r, file))

query_corpus = []
query_names = []
# iterate throught the query database list in sorted manner
for file in tqdm(sorted(query_file_paths), desc="query documents"):
    
    open_file = open(file, 'r', encoding="utf-8")
    text = open_file.read()
    str_list = text.splitlines()
    query_corpus.append(str_list)
    query_names.append(os.path.basename(file))
    open_file.close()

score_dict = {}
prediction_dict = {}
pred_df = pd.DataFrame(columns=['Documend id', 'Noticed cases', 'Numbers of noticed cases', 'Prediction list'])
for i in tqdm(range(len(query_corpus))):
    query_name = query_names[i]
    print(query_name)
    que_text = query_corpus[i][0]
    doc_scores = bm25.transform(que_text, corpus)
    rev_doc_score = sorted(doc_scores, reverse=True)
    score_dict[query_name] = doc_scores
    doc_sort_index = np.argsort(doc_scores)
    do_sort_index_rev = doc_sort_index[::-1]
    prediction_dict[query_name] = do_sort_index_rev
    predictions = [citation_names[case] for case in prediction_dict[query_name]]

    pred_df = pred_df.append({'Documend id': query_name, 'Noticed cases': noticed_case_list[query_name],'Numbers of noticed cases': len(noticed_case_list[query_name]), 'Prediction list': predictions},ignore_index=True)

# create rsults directory
pred_df.to_csv('./bm25_test_2023_ngarm('+ str(args.ngram_1) + str(args.ngram_2) +').csv')
## bm25 computing ends

## Evaluation starts
print('ngram:', str(args.ngram_1), str(args.ngram_2))
file_path = './bm25_test_2023_ngarm('+ str(args.ngram_1) + str(args.ngram_2) +').csv'
with open(file_path, 'r')as csvfile:
    pred_df = pd.read_csv(csvfile, delimiter=',')
    csvfile.close()

correct_pred = 0
retri_cases = 0
relevant_cases = 0
for i in tqdm(pred_df.index):
    query_case = pred_df.iloc[i, 1]
    true_list = noticed_case_list[query_case]
    pred_list_0 = pred_df.iloc[i, 4]
    pred_list = pred_list_0.split("['")[1].split("']")[0].split("', '")
    
    # top-k prediction
    k = args.topk

    c_p, r_c = micro_prec(true_list, pred_list, k)
    correct_pred += c_p
    retri_cases += r_c
    relevant_cases += len(true_list)

M_pre = correct_pred/retri_cases
M_recall = correct_pred/relevant_cases
M_F = 2*M_pre*M_recall/ (M_pre + M_recall)


print('Evaluation')
print("Correct Predictions: ", correct_pred)
print("Retrived Cases: ", retri_cases)
print("Noticed Cases: ", relevant_cases)


print("Precision score: ", M_pre)
print("Recall score: ", M_recall)
print("F1 score: ", M_F)
## Evaluation ends

print('Finished!')