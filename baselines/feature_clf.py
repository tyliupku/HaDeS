#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 下午9:43
# @Author  : Tianyu Liu
from transformers import BertTokenizer, BertForMaskedLM, AutoModelWithLMHead, AutoTokenizer, BertConfig, AutoConfig
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from nltk import sent_tokenize
from nltk.corpus import stopwords
import random, os
import torch.nn as nn
import torch.nn.functional as F
import codecs
import argparse
import spacy
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as color
import matplotlib.pyplot as plt
from nlp import load_dataset
from collections import defaultdict
import json
import pandas as pd
import seaborn as sns
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, \
    f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
from utils import binary_eval

def remove_marked_sen(sen, start_id, end_id):
    tokens = sen.strip().split()
    assert tokens[start_id].startswith("===") and tokens[end_id].endswith("===")
    tokens[start_id] = tokens[start_id][3:]
    tokens[end_id] = tokens[end_id][:-3]
    return tokens


def get_ppmi_matrix(voc, path="../data_collections/Wiki-Hades/train.txt"):
    def co_occurrence(sentences, window_size):
        d = defaultdict(int)
        vocab = set(voc)
        for text in sentences:
            # iterate over sentences
            # print(text)
            for i in range(len(text)):
                token = text[i]
                next_token = text[i+1 : i+1+window_size]
                for t in next_token:
                    if t in vocab and token in vocab:
                        key = tuple( sorted([t, token]) )
                        d[key] += 1
        # print(vocab)
        print(len(vocab))

        # formulate the dictionary into dataframe
        vocab = sorted(vocab) # sort vocab
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                          index=vocab,
                          columns=vocab)
        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value
        return df

    def pmi(df, positive=True):
        col_totals = df.sum(axis=0)
        total = col_totals.sum()
        row_totals = df.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        df = df / expected
        '''
        # Silence distracting warnings about log(0):
        print("taking LOG ...")
        with np.errstate(divide='ignore'):
            df = np.log(df)
        df[np.isinf(df)] = 0.0  # log(0) = 0
        print("Turn positive ...")
        if positive:
            df[df < 0] = 0.0
        '''
        return df

    corpus = []

    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            tgt, tgt_ids = example["replaced"], example["replaced_ids"]
            sen = remove_marked_sen(tgt, tgt_ids[0], tgt_ids[1])
            corpus.append(sen)

    df = co_occurrence(corpus, 1)
    ppmi = pmi(df, positive=True)
    print("finish")
    return ppmi


def get_idf_matrix(path="../data_collections/Wiki-Hades/train.txt"):
    corpus = []

    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            tgt, tgt_ids = example["replaced"], example["replaced_ids"]
            sen = remove_marked_sen(tgt, tgt_ids[0], tgt_ids[1])
            corpus.append(" ".join(sen))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    word = vectorizer.get_feature_names()
    num_doc, num_vocab = X.shape
    X = np.array(X>0, dtype=int)
    word_idf = np.log10(num_doc / (X.sum(0)+1))
    idf_dic = dict()
    for w, idf in zip(word, word_idf):
        idf_dic[w] = idf

    word_freq = X.sum(0)
    word_freq = word_freq / word_freq.sum()

    return idf_dic, word_freq


def read_file(path="wiki5k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate.finish"):
    labels = {0:0, 1:0, 2:0}
    lineno = 1
    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            labels[example["hallucination"]] += 1
            lineno += 1
    print("Total: {}".format(lineno))
    print(labels)


def subsets(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    ans = []
    def dfs(curpos, tmp):
        if tmp:
            ans.append(tmp[:])
        for i in range(curpos, len(nums)):
            tmp.append(nums[i])
            dfs(i+1, tmp)
            tmp.pop(-1)
    dfs(0, [])
    return ans


class ClfModel:
      def __init__(self, args):
            self.idf_dic, self.p_word = get_idf_matrix()
            if not os.path.exists("ppmi.pkl"):
                print("reading ppmi ...")
                word_ppmi = get_ppmi_matrix(list(self.idf_dic.keys())[:])
                self.word_ppmi = word_ppmi
                word_ppmi.to_pickle("ppmi.pkl")
            else:
                word_ppmi = pd.read_pickle("ppmi.pkl")
                self.word_ppmi = word_ppmi
            self.args = args
            self.device = args.device
            self.rep_model = AutoModelWithLMHead.from_pretrained("bert-base-uncased").to(self.device)
            self.rep_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # self.clf = svm.LinearSVC()
            # self.clf = LogisticRegression(random_state=0)
            self.clf = make_pipeline(StandardScaler(),
                                     SGDClassifier(loss="log", max_iter=10000, tol=1e-5))

      def get_ppmi_features(self, src_sen, rep_sen, src_ids, rep_ids):
            rep_tokens = rep_sen.strip().split()
            rep_start_id, rep_end_id = rep_ids
            assert rep_tokens[rep_start_id].startswith("===") and \
                   rep_tokens[rep_end_id].endswith("===")
            rep_tokens[rep_start_id] = rep_tokens[rep_start_id][3:]
            rep_tokens[rep_end_id] = rep_tokens[rep_end_id][:-3]

            max_ppmi, mean_ppmi, min_ppmi = [], [], []
            for idx in range(rep_start_id, rep_end_id+1):
                ppmis = []
                for j in range(rep_start_id):
                    v = 0
                    if rep_tokens[idx] in self.word_ppmi.columns and \
                       rep_tokens[j] in self.word_ppmi.columns:
                        v = self.word_ppmi.at[rep_tokens[idx], rep_tokens[j]]
                        if v > 0:
                            v = max(0, np.log(v))
                        else:
                            v = 0 # not gona happen
                    ppmis.append(v)
                max_ppmi.append(max(ppmis))
                min_ppmi.append(min(ppmis))
                mean_ppmi.append(sum(ppmis)/len(ppmis))

            return sum(mean_ppmi) / len(mean_ppmi), sum(max_ppmi) / len(max_ppmi), sum(min_ppmi) / len(min_ppmi)

      def get_tfidf_features(self, src_sen, rep_sen, src_ids, rep_ids):
            rep_tokens = rep_sen.strip().split()
            rep_start_id, rep_end_id = rep_ids
            assert rep_tokens[rep_start_id].startswith("===") and \
                   rep_tokens[rep_end_id].endswith("===")
            rep_tokens[rep_start_id] = rep_tokens[rep_start_id][3:]
            rep_tokens[rep_end_id] = rep_tokens[rep_end_id][:-3]

            #  TF-IDF features
            tf_dic = dict()
            for token in rep_tokens:
                if token in self.idf_dic:
                    if token not in tf_dic:
                        tf_dic[token] = 1
                    else:
                        tf_dic[token] += 1

            tf_total = sum(tf_dic.values())
            tfidf_list = []
            for idx in range(rep_start_id, rep_end_id+1):
                if rep_tokens[idx] in self.idf_dic:
                    tfidf_list.append(tf_dic[rep_tokens[idx]] * self.idf_dic[rep_tokens[idx]] / tf_total)
            tfidf_max = max(tfidf_list) if tfidf_list else 0.
            tfidf_min = min(tfidf_list) if tfidf_list else 0.
            tfidf_mean = sum(tfidf_list)/len(tfidf_list) if tfidf_list else 0.
            return tfidf_mean, tfidf_max, tfidf_min

      def encode_bert(self, src_sen, rep_sen, src_ids, rep_ids):
            rep_tokens = rep_sen.strip().split()
            rep_start_id, rep_end_id = rep_ids
            assert rep_tokens[rep_start_id].startswith("===") and \
                   rep_tokens[rep_end_id].endswith("===")
            rep_tokens[rep_start_id] = rep_tokens[rep_start_id][3:]
            rep_tokens[rep_end_id] = rep_tokens[rep_end_id][:-3]

            #  Prob, Entropy features
            rep_subtokens = ["[CLS]"]
            tokenizer = self.rep_tokenizer
            model = self.rep_model
            rep_mask_start_id, rep_mask_end_id = 0, 0
            for id, rep_token in enumerate(rep_tokens):
                rep_subtoken = tokenizer.tokenize(rep_token)
                if id == rep_start_id:
                    rep_mask_start_id = len(rep_subtokens)
                if id == rep_end_id:
                    rep_mask_end_id = len(rep_subtokens) + len(rep_subtoken)
                if id >= rep_start_id and id <= rep_end_id:
                    rep_subtokens.extend(len(rep_subtoken) * ["[MASK]"])
                else:
                    rep_subtokens.extend(rep_subtoken)
            rep_subtokens.append("[SEP]")
            rep_input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(rep_subtokens)).unsqueeze(0).to(self.device)
            # print("rep_input_ids {}".format(rep_input_ids.size()))
            prediction_scores = model(rep_input_ids)[0]
            # print(prediction_scores)
            # print(prediction_scores.sum())
            # print(prediction_scores.size())
            prediction_scores = F.softmax(prediction_scores, dim=-1)
            # print("prediction_scores {}".format(prediction_scores.size()))
            # print("mask_start {}  mask_end {}".format(rep_mask_start_id, rep_mask_end_id))

            scores = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                subtoken_score = prediction_scores[0, id, rep_input_ids[0][id]].item()
                scores.append(subtoken_score)

            entropies = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                vocab_scores = prediction_scores[0, id].detach().cpu().numpy()
                entropy = np.sum(np.log(vocab_scores+1e-11) * vocab_scores)
                entropies.append(-entropy)

            return sum(scores)/len(scores), max(scores), min(scores), \
                   sum(entropies)/len(entropies), max(entropies), min(entropies)

      def encode_gpt(self, src_sen, rep_sen, src_ids, rep_ids):
            rep_tokens = rep_sen.strip().split()
            rep_start_id, rep_end_id = rep_ids
            assert rep_tokens[rep_start_id].startswith("===") and \
                   rep_tokens[rep_end_id].endswith("===")
            rep_tokens[rep_start_id] = rep_tokens[rep_start_id][3:]
            rep_tokens[rep_end_id] = rep_tokens[rep_end_id][:-3]
            rep_subtokens = []
            tokenizer = self.rep_tokenizer
            model = self.rep_model
            rep_mask_start_id, rep_mask_end_id = 0, 0
            for id, rep_token in enumerate(rep_tokens):
                rep_token = " "+rep_token if id!=0 else rep_token
                rep_subtoken = tokenizer.tokenize(rep_token)
                if id == rep_start_id:
                    rep_mask_start_id = len(rep_subtokens)
                if id == rep_end_id:
                    rep_mask_end_id = len(rep_subtokens) + len(rep_subtoken)

                rep_subtokens.extend(rep_subtoken)

            rep_input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(rep_subtokens)).unsqueeze(0).to(self.device)
            # print("rep_input_ids {}".format(rep_input_ids.size()))
            prediction_scores = model(rep_input_ids)[0]
            prediction_scores = F.softmax(prediction_scores, dim=-1)
            # print("prediction_scores {}".format(prediction_scores.size()))
            # print("mask_start {}  mask_end {}".format(rep_mask_start_id, rep_mask_end_id))

            scores = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                subtoken_score = prediction_scores[0, id, rep_input_ids[0][id]].item()
                scores.append(subtoken_score)

            entropies = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                vocab_scores = prediction_scores[0, id].detach().cpu().numpy()
                entropy = np.sum(np.log(vocab_scores+1e-11) * vocab_scores)
                entropies.append(-entropy)

            return sum(scores)/len(scores), max(scores), \
                   sum(entropies)/len(entropies), max(entropies)

      def feature_correlation(self, trainpath="wiki5k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate.finish",
              testpath="selected_replaced_instance.annotate.finish"):
            def draw_pic(dataframe, savename):
                plt.figure(figsize=(10, 8))
                heatmap = sns.heatmap(dataframe, vmin=-1, vmax=1, annot=True, cmap='BrBG')
                heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
                # save heatmap as .png file
                # dpi - sets the resolution of the saved image in dots/inches
                # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
                plt.savefig(savename, dpi=300, bbox_inches='tight')

            feature_names = {"avgscore": 0, "avgentro": 1, "avgtfidf": 2, "avgppmi": 3,
                             "maxscore": 4, "maxentro": 5, "maxtfidf": 6, "maxppmi": 7}
            feature_keys = list(feature_names.keys())[:]
            combinations = subsets(feature_keys)

            encode_func = self.encode_gpt if "gpt" in self.args.rep_model else self.encode_bert
            trainx, trainy = [], []
            print("Load Training Features ...")
            with codecs.open(trainpath, "r", encoding="utf-8") as fr:
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    avgscore, maxscore, avgentro, maxentro = encode_func(example["src"], example["replaced"],
                                                              example["src_ids"], example["replaced_ids"])
                    avgtfidf, maxtfidf = self.get_tfidf_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    avgppmi, maxppmi = self.get_ppmi_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    features = [avgscore, avgentro, avgtfidf, avgppmi,
                                maxscore, maxentro, maxtfidf, maxppmi]

                    trainx.append(features+[label])
                    trainy.append(label)

            feature_names = ["avgscore", "avgentro", "avgtfidf", "avgppmi",
                             "maxscore", "maxentro", "maxtfidf", "maxppmi", "label"]

            train_df = pd.DataFrame(np.array(trainx), columns=feature_names)
            draw_pic(train_df.corr(), "train_correlation.png")
            fw = codecs.open("0errors.txt", "w+", encoding="utf-8")
            testx, testy = [], []
            with codecs.open(testpath, "r", encoding="utf-8") as fr:
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    avgscore, maxscore, avgentro, maxentro = encode_func(example["src"], example["replaced"],
                                                              example["src_ids"], example["replaced_ids"])
                    avgtfidf, maxtfidf = self.get_tfidf_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    avgppmi, maxppmi = self.get_ppmi_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    features = [avgscore, avgentro, avgtfidf, avgppmi,
                                maxscore, maxentro, maxtfidf, maxppmi]

                    if label == 0:
                        fw.write(example["replaced"]+"\n\n"+
                                 "avgscore {} ; avgentro {} ; avgtfidf {} ; avgppmi {}".format(avgscore, avgentro,
                                                                                               avgtfidf, avgppmi)+"\n\n")
                    testx.append(features+[label])
                    testy.append(label)
            test_df = pd.DataFrame(np.array(testx), columns=feature_names)
            draw_pic(test_df.corr(), "test_correlation.png")

      def append_features_to_file(self, prefix="../human_annotation/uhrs_src/", frname="data_merge_tag1.txt",
                                  fwname="data_merge_finish.txt"):
            encode_func = self.encode_gpt if "gpt" in self.args.rep_model else self.encode_bert
            fw = codecs.open(os.path.join(prefix, fwname), "w+", encoding="utf-8")
            cnt = 0
            with codecs.open(os.path.join(prefix, frname), "r", encoding="utf-8") as fr:
                for line in fr:
                    example = json.loads(line.strip())
                    src, tgt, src_ids, tgt_ids = example["src"], example["replaced"], \
                                                 example["src_ids"], example["replaced_ids"]
                    avgscore, maxscore, minscore, avgentro, maxentro, minentro = encode_func(src, tgt, src_ids, tgt_ids)
                    avgtfidf, maxtfidf, mintfidf = self.get_tfidf_features(src, tgt, src_ids, tgt_ids)
                    avgppmi, maxppmi, minppmi = self.get_ppmi_features(src, tgt, src_ids, tgt_ids)
                    newdic = {"avgprob":float(avgscore), "maxprob":float(maxscore), "minprob":float(minscore),
                                "avgentro":float(avgentro), "maxentro":float(maxentro), "minentro":float(minentro),
                                "avgtfidf":float(avgtfidf), "maxtfidf":float(maxtfidf), "mintfidf":float(mintfidf),
                                "avgppmi":float(avgppmi), "maxppmi":float(maxppmi), "minppmi":float(minppmi)}
                    # print(newdic)
                    # print([(name, type(val)) for name, val in newdic.items()])
                    example.update(newdic)

                    fw.write(json.dumps(example)+"\n")
                    cnt += 1
                    if cnt % 100==0:
                        print(cnt)

      def train(self, trainpath="../data_collections/Wiki-Hades/train.txt",
              # testpath="selected_replaced_instance.annotate.finish", ):
              testpath="../data_collections/Wiki-Hades/test.txt", epoch=10):
            feature_names = {"avgscore": 0, "avgentro": 1, "avgtfidf": 2, "avgppmi": 3,
                             "maxscore": 4, "maxentro": 5, "maxtfidf": 6, "maxppmi": 7}
            feature_keys = list(feature_names.keys())[:]
            combinations = subsets(feature_keys)

            encode_func = self.encode_gpt if "gpt" in self.args.rep_model else self.encode_bert
            trainx, trainy = [], []
            print("Load Training Features ...")
            with codecs.open(trainpath, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    avgscore, maxscore, _, avgentro, maxentro, _ = encode_func(example["src"], example["replaced"],
                                                              example["src_ids"], example["replaced_ids"])
                    avgtfidf, maxtfidf, _ = self.get_tfidf_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    avgppmi, maxppmi, _ = self.get_ppmi_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    features = [avgscore, avgentro, avgtfidf, avgppmi,
                                maxscore, maxentro, maxtfidf, maxppmi]

                    trainx.append(features)
                    trainy.append(label)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("Train {}".format(cnt))
            print(trainx[:30])

            testx, testy = [], []
            with codecs.open(testpath, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    avgscore, maxscore, _, avgentro, maxentro, _ = encode_func(example["src"], example["replaced"],
                                                              example["src_ids"], example["replaced_ids"])
                    avgtfidf, maxtfidf, _ = self.get_tfidf_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    avgppmi, maxppmi, _ = self.get_ppmi_features(example["src"], example["replaced"],
                                                                 example["src_ids"], example["replaced_ids"])
                    features = [avgscore, avgentro, avgtfidf, avgppmi,
                                maxscore, maxentro, maxtfidf, maxppmi]

                    testx.append(features)
                    testy.append(label)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("Test {}".format(cnt))

            fw = codecs.open("feature_combination.txt", "w+", encoding="utf-8")
            infos = []
            for feats in combinations:
                real_trainx = []
                for fs in trainx:
                    new_fs = []
                    for featname in feats:
                        new_fs.append(fs[feature_names[featname]])
                    real_trainx.append(new_fs)
                real_testx = []
                for fs in testx:
                    new_fs = []
                    for featname in feats:
                        new_fs.append(fs[feature_names[featname]])
                    real_testx.append(new_fs)

                self.clf.fit(real_trainx, trainy)
                predy = self.clf.predict(real_testx)

                print("="*20)
                print("Features: {}".format(" ".join(feats)))
                feat_str = "Features: {}".format(" ".join(feats))
                # fw.write("\n\nFeatures: {}\n".format(" ".join(feats)))
                acc, info = binary_eval([0]*len(testy), testy, return_f1=False)
                infos.append([acc, feat_str, info])
            infos = sorted(infos, key=lambda x:-x[0])

            for item in infos:
                _, feat_str, info = item
                fw.write("\n\n"+feat_str+"\n")
                fw.write(info+"\n")

      def stat_show(self, path="wiki5k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate.finish",
              stat_pos_ner=False):
            lineno = 0
            avgscores, maxscores, avgentros, maxentros = [], [], [], []
            encode_func = self.encode_gpt if "gpt" in self.args.rep_model else self.encode_bert
            with codecs.open(path, "r", encoding="utf-8") as fr:
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label != 2:
                        avgscore, maxscore, avgentro, maxentro = encode_func(example["src"], example["replaced"],
                                                              example["src_ids"], example["replaced_ids"])
                        avgscores.append((avgscore, label))
                        maxscores.append((maxscore, label))
                        avgentros.append((avgentro, label))
                        maxentros.append((maxentro, label))
                        if stat_pos_ner:
                            self.ner_pos_stat(example["src"], example["replaced"],
                                              example["src_ids"], example["replaced_ids"], label)
                    lineno += 1
                    print(lineno)

            def show_statistics(avgscores):
                avgscores = sorted(avgscores, key=lambda x:-x[0])
                tot_hallu_num = sum([avgscore[1] for avgscore in avgscores])
                tot_clear_num = len(avgscores) - tot_hallu_num
                ratio_hallu = 1.0 * sum(avgscore[1] for avgscore in avgscores[:tot_hallu_num]) / tot_hallu_num
                ratio_clear = 1.0 * (tot_clear_num-sum(avgscore[1] for avgscore in avgscores[:tot_clear_num])) / tot_clear_num
                # print("Higher Prob means more hallucination {}".format(ratio_hallu))
                # print("higher Prob means less hallucination {}".format(ratio_clear))

                avg_hallu_prob = sum([avgscore[0] for avgscore in avgscores if avgscore[1]==1]) / tot_hallu_num
                avg_clear_prob = sum([avgscore[0] for avgscore in avgscores if avgscore[1]==0]) / tot_clear_num
                assert len([avgscore[0] for avgscore in avgscores if avgscore[1]==1]) == tot_hallu_num
                assert len([avgscore[0] for avgscore in avgscores if avgscore[1]==0]) == tot_clear_num
                print("Hallucination Prob : {} : Clear Prob : {}".format(avg_hallu_prob, avg_clear_prob))

            if stat_pos_ner:
                for ner in self.total_ner_stats:
                    print("{} : {:.4f}".format(ner, 1.0 * self.hallu_ner_stats[ner] / self.total_ner_stats[ner]))
                for pos in self.total_pos_stats:
                    print("{} : {:.4f}".format(pos, 1.0 * self.hallu_pos_stats[pos] / self.total_pos_stats[pos]))

            print("Average Prob:")
            show_statistics(avgscores)
            print("Max Prob:")
            show_statistics(maxscores)
            print("Average Entropy:")
            show_statistics(avgentros)
            print("Max Entropy:")
            show_statistics(maxentros)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", default=10, type=int) # topk first
    parser.add_argument("--topp", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--bert_layer", default=10, type=int)
    parser.add_argument("--bert_mask", action="store_true")
    parser.add_argument("--num_epoch", default=20, type=int)
    parser.add_argument("--params", default="bert", type=str)
    parser.add_argument("--load_ft_gpt", action="store_true")
    parser.add_argument("--load_model", default="../finetune_learn/output", type=str)
    parser.add_argument("--context_len", default=1, type=int)
    parser.add_argument("--replace_ratio", default=0.6, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--min_replace_time", default=0, type=int)
    parser.add_argument("--sample_num", default=1, type=int)
    parser.add_argument("--context_limit", default=0, type=int)
    parser.add_argument("--mode_name", default="len-fix", type=str)
    parser.add_argument("--rep_model", default="bert", type=str)
    parser.add_argument("--rank_model", default="gpt2", type=str)
    parser.add_argument("--ignore_oracle", action="store_true")
    parser.add_argument("--parallel_decoding", action="store_true")
    parser.add_argument("--mask_before_replacement", action="store_true")
    args = parser.parse_args()

    # ppmi = get_ppmi_matrix()
    # print(ppmi.columns)
    # print(len(ppmi.columns))
    rep_op = ClfModel(args)
    # rep_op.feature_analysis()
    # rep_op.append_features_to_file()
    rep_op.train()
    # rep_op.run(path="selected_replaced_instance.annotate.finish")

    '''
    learning_rate0 = args.lr
    weight_decay_finetune = 1e-5

    if "all" in args.params:
        named_params = list(rep_op.hidden2label.named_parameters()) + \
                       list(rep_op.rep_model.named_parameters())
    else:
        named_params = list(rep_op.hidden2label.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optim_func = torch.optim.Adam if "gpt" in args.rep_model else BertAdam
    optimizer = optim_func(optimizer_grouped_parameters, lr=learning_rate0)
    loss = nn.CrossEntropyLoss().to(args.device)
    rep_op.bert_run(optimizer, loss, epoch=args.num_epoch)
    '''
