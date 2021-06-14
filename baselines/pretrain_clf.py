from transformers import BertTokenizer, BertForMaskedLM, AutoModelWithLMHead, AutoTokenizer, BertConfig, AutoConfig, \
    XLNetLMHeadModel, DebertaTokenizer, DebertaModel
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import random, os, sys
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
import csv
import pandas as pd
import seaborn as sns
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from torch.utils import data
from data_loader import DataProcessor, HalluDataset, get_examples_from_sen_tuple, example2feature
from utils import binary_eval, read_file, subsets, sent_ner_bounds, show_output_size
from torch.utils.tensorboard import SummaryWriter



class ClfModel(nn.Module):
      def __init__(self, args):
            super().__init__()


            if args.rep_model == "bert":
                rep_model_name = 'bert-large-uncased'
            elif args.rep_model == "bert-small":
                rep_model_name = 'bert-base-uncased'
            elif args.rep_model == "roberta":
                rep_model_name = 'roberta-large'
            elif args.rep_model == "roberta-mnli":
                rep_model_name = 'roberta-large-mnli'
            elif args.rep_model == "roberta-small":
                rep_model_name = 'roberta-base'
            elif args.rep_model == "bart":
                rep_model_name = 'facebook/bart-large'
            elif args.rep_model == "t5":
                rep_model_name = 't5-base'
            elif args.rep_model == "distillbert":
                rep_model_name = 'distilbert-base-uncased'
            elif args.rep_model == "distillroberta":
                rep_model_name = "distilroberta-base"
            elif args.rep_model == "longformer":
                rep_model_name = "allenai/longformer-base-4096"
            elif args.rep_model == "albert":
                rep_model_name = 'albert-base-v1'
            elif args.rep_model == "albertv2":
                rep_model_name = 'albert-base-v2'
            elif args.rep_model == "albert-large":
                rep_model_name = 'albert-xlarge-v1'
            elif args.rep_model == "xlnet-small":
                rep_model_name = 'xlnet-base-cased'
            elif args.rep_model == "xlnet":
                rep_model_name = 'xlnet-large-cased'
            elif args.rep_model == "deberta":
                rep_model_name = 'microsoft/deberta-base'
            elif args.rep_model == "gpt2":
                rep_model_name = 'gpt2-medium'
            else:
                rep_model_name = 'gpt2'

            if args.rank_model == "bert":
                rank_model_name = 'bert-large-uncased'
            else: rank_model_name = 'gpt2'
            self.rep_name = args.rep_model

            if args.rep_model == "deberta":
                self.rep_tokenizer = DebertaTokenizer.from_pretrained(rep_model_name)
                self.rep_model = DebertaModel.from_pretrained(rep_model_name).to(args.device)
            elif "xlnet" in args.rep_model:
                self.rep_tokenizer = AutoTokenizer.from_pretrained(rep_model_name)
                self.rep_model = XLNetLMHeadModel.from_pretrained(rep_model_name, mem_len=1024).to(args.device)
            else:
                self.rep_tokenizer = AutoTokenizer.from_pretrained(args.load_model\
                                                                   if rep_model_name=="gpt" and args.load_ft_gpt
                                                                   else rep_model_name)


                config = AutoConfig.from_pretrained(args.load_model \
                                                    if rep_model_name=="gpt" and args.load_ft_gpt
                                                    else rep_model_name)
                # config.return_dict = True
                config.output_hidden_states = True
                print(config)
                self.rep_model = AutoModelWithLMHead.from_pretrained(args.load_model \
                                                       if rep_model_name=="gpt" and args.load_ft_gpt
                                                       else rep_model_name, config=config).to(args.device)

            self.rank_name = args.rank_model
            self.rank_tokenizer = AutoTokenizer.from_pretrained(rank_model_name)
            self.rank_model = AutoModelWithLMHead.from_pretrained(rank_model_name).to(args.device)

            self.dropout = torch.nn.Dropout(0.2)
            hidden_size = 1024 if "large" in rep_model_name or rep_model_name=="gpt2-medium" else 768
            if args.mode == 1:
                self.hidden2label = nn.Linear(hidden_size, 2).to(args.device)
            elif args.mode == 2:
                self.hidden2label = nn.Sequential(
                                        nn.Linear(hidden_size, hidden_size//2),
                                        nn.Sigmoid(),
                                        nn.Linear(hidden_size//2, 2)).to(args.device)
            self.dropout = torch.nn.Dropout(args.dropout)
            self.layer = args.bert_layer
            self.bert_mask = args.bert_mask

            self.eval()
            self.eval()
            self.device = args.device
            self.args = args

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

      def bert_run(self, optim, trainpath="../data_collections/Wiki-Hades/train.txt",
                   testpath="../data_collections/Wiki-Hades/test.txt",
                   validpath="../data_collections/Wiki-Hades/valid.txt",
                   epoch=10):
              # testpath="selected_replaced_instance.annotate.finish", epoch=10):
            prefix = "runs/{}_{}_lr_{}_dp_{}_mode{}_{}_clen{}/".format(self.rep_name, self.args.name, self.args.lr,
                                            self.args.dropout, self.args.mode, self.args.task_mode, self.args.clen)
            bestmodelpath = prefix + "best_model.pt"
            epoch_start = 1
            if os.path.exists(bestmodelpath) and self.args.continue_train:
                checkpoint = torch.load(bestmodelpath)
                self.load_state_dict(checkpoint["model_state_dict"])
                epoch_start = checkpoint["epoch"] + 1

            writer = SummaryWriter(prefix)
            csvlogger = prefix + "log.csv"
            txtlogger = prefix + "test_log.txt"
            txtwriter = codecs.open(txtlogger, "w+")
            vtxtlogger = prefix + "valid_log.txt"
            vtxtwriter = codecs.open(vtxtlogger, "w+")

            if not os.path.exists(csvlogger):
                csvfile = open(csvlogger, 'w+')
                fileHeader = ["epoch", "H_p", "H_r", "H_f1", "H_gmean", "C_p", "C_r", "C_f1", "C_gmean",
                              "Acc", "BSS", "ROC_AUC", "AP_AUC"]
                csvwrite = csv.writer(csvfile)
                csvwrite.writerow(fileHeader)
            else:
                csvfile = open(csvlogger, 'a')
                csvwrite = csv.writer(csvfile)

            dp = DataProcessor()
            train_examples = dp.get_examples(trainpath)

            train_dataset = HalluDataset(train_examples, self.rep_tokenizer, self.args.clen,
                                         self.rep_name, self.args.task_mode)

            train_dataloader = data.DataLoader(dataset=train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)
            nSamples = dp.get_label_dist()
            print("====Train label : {}".format(nSamples))
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]

            test_examples = dp.get_examples(testpath)
            test_dataset = HalluDataset(test_examples, self.rep_tokenizer, self.args.clen,
                                        self.rep_name, self.args.task_mode)
            test_dataloader = data.DataLoader(dataset=test_dataset,
                                               batch_size=4,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)

            valid_examples = dp.get_examples(validpath)
            valid_dataset = HalluDataset(valid_examples, self.rep_tokenizer, self.args.clen,
                                         self.rep_name, self.args.task_mode)
            valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                               batch_size=4,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)

            normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
            loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
            # loss_func = nn.CrossEntropyLoss().to(self.args.device)
            fwd_func = self.bert_train # if "gpt" not in self.args.rep_model else self.gpt_train
            best_acc = -1
            best_f1_score = -1
            for ei in range(epoch_start, epoch+1):
                cnt = 0
                self.train()
                train_loss = 0
                predy, trainy, hallu_sm_score = [], [], []
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(self.device) for t in batch[:-1])
                    input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                    score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    # print("pred {}".format(pred.size()))
                    # print(label_ids.tolist())
                    # print(pred.tolist())
                    trainy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    # print(loss)
                    train_loss += loss.item()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    cnt += 1
                    if cnt % 10 == 0:
                        print("Training Epoch {} - {} - {}".format(ei, cnt, train_loss/cnt))
                print("Training Epoch {} ...".format(ei))
                # print(predy)
                acc, (f1, precision, recall, gmean), (bss, roc_auc, ap_auc), _ = \
                    binary_eval(predy, trainy, return_f1=True, predscore=hallu_sm_score)
                writer.add_scalar('Loss/train_epoch', train_loss, ei)
                writer.add_scalar('F1/train_consistent_epoch', f1[0], ei)
                writer.add_scalar('Precision/train_consistent_epoch', precision[0], ei)
                writer.add_scalar('Recall/train_consistent_epoch', recall[0], ei)
                writer.add_scalar('F1/train_hallucination_epoch', f1[1], ei)
                writer.add_scalar('Precision/train_hallucination_epoch', precision[1], ei)
                writer.add_scalar('Recall/train_hallucination_epoch', recall[1], ei)
                writer.add_scalar('Acc/train_epoch', acc, ei)
                print("Train Epoch {} end ! Loss : {}".format(ei, train_loss))

                if ei % 21 == 0:
                    savemodel_path = prefix + "model_{}_{}_{}.pt".format(ei, f1[0], f1[1])
                    torch.save(
                    {"model_state_dict": self.state_dict(),
                     "optim_state_dict": optim.state_dict(),
                     "train_f1": f1,
                     "train_precision": precision,
                     "train_recall": recall,
                     "train_acc": acc,
                     "epoch": epoch},
                     savemodel_path)
                    # self.get_model_scores(savemodel_path, epoch=ei)

                storage = []
                self.eval()
                predy, testy, hallu_sm_score = [], [], []
                test_loss = 0
                for step, batch in enumerate(test_dataloader):
                    batch = tuple(t.to(self.device) for t in batch[:-1])
                    input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                    score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    testy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    test_loss += loss.item()
                print("Testing Epoch {} ...".format(ei))

                acc, (f1, precision, recall, gmean), (bss, roc_auc, ap_auc), info = \
                    binary_eval(predy, testy, return_f1=True, predscore=hallu_sm_score)
                writer.add_scalar('Loss/test_epoch', test_loss, ei)
                writer.add_scalar('F1/test_consistent_epoch', f1[0], ei)
                writer.add_scalar('Precision/test_consistent_epoch', precision[0], ei)
                writer.add_scalar('Recall/test_consistent_epoch', recall[0], ei)
                writer.add_scalar('F1/test_hallucination_epoch', f1[1], ei)
                writer.add_scalar('Precision/test_hallucination_epoch', precision[1], ei)
                writer.add_scalar('Recall/test_hallucination_epoch', recall[1], ei)
                writer.add_scalar('Acc/test_epoch', acc, ei)

                csvfile = open(csvlogger, 'a')
                csvwrite = csv.writer(csvfile)
                rowdata = [ei, precision[1], recall[1], f1[1], gmean[1], precision[0], recall[0], f1[0], gmean[0],\
                            acc, bss, roc_auc, ap_auc]
                rowdata = [str(f) for f in rowdata]
                csvwrite.writerow(rowdata)
                txtwriter.write("Epoch {}\n".format(ei) + info + "\n\n")

                f1_score = f1[0] + f1[0]
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    torch.save({"model_state_dict": self.state_dict(),
                                "optim_state_dict": optim.state_dict(),
                                "test_f1": f1,
                                "test_precision": precision,
                                "test_recall": recall,
                                "test_acc": acc,
                                "epoch": epoch},
                                prefix + "best_model.pt")

                self.eval()
                predy, testy, hallu_sm_score = [], [], []
                valid_loss = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.to(self.device) for t in batch[:-1])
                    input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                    score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    testy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    valid_loss += loss.item()
                print("Valid Epoch {} ...".format(ei))

                acc, (f1, precision, recall, gmean), (bss, roc_auc, ap_auc), info = \
                    binary_eval(predy, testy, return_f1=True, predscore=hallu_sm_score)
                vtxtwriter.write("Epoch {}\n".format(ei) + info + "\n\n")

                if acc > best_acc:
                    best_acc = acc
                    storage = sorted(storage, key=lambda x:-x[1])
                    fw = codecs.open("200cases_from_model.json", "w+", encoding="utf-8")
                    cnt = 1
                    for i in range(len(storage)):
                        example = storage[i][-1]
                        example["score"] = storage[i][1]
                        if "UNK" in example["replaced"]:
                            continue
                        example["sample_from"] = "bert_large_confidence"
                        fw.write(json.dumps(example)+"\n")
                        if cnt > 200:
                            break
                        cnt += 1

            '''
            checkpoint = torch.load("runs/enrich_bert_lr_{}_dp{}".format(self.args.lr, self.args.dropout))
            self.rep_model.load_state_dict(checkpoint["model_state_dict"])
            self.rep_model.eval()
            fwname = unannotatedpath + ".modelscore"
            with codecs.open(fwname, "w+", encoding="utf-8") as fw:
                pass
            '''

      def sent_hallu_vis(self, sen, modelpath="runs/datam4424_lr_0.005_dp0.2/model_9_0.5991348233597694_0.8169245966414226.pt"):
            checkpoint = torch.load(modelpath)
            tsen, rep_pos = sent_ner_bounds(sen, self.nlp)
            print("tsen {} rep_pos {}".format(len(tsen), len(rep_pos)))
            examples = get_examples_from_sen_tuple(tsen, rep_pos)
            print("len of example {}".format(len(examples)))
            # examples = examples[30:]
            print(examples[0].idxs)
            print(examples[0].sen)
            print(len(examples))
            unannotated_dataset = HalluDataset(examples, self.rep_tokenizer, 200, self.rep_name)
            print(unannotated_dataset[0])
            print(len(unannotated_dataset))
            unannotated_dataloader = data.DataLoader(dataset=unannotated_dataset,
                                                       batch_size=8,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       collate_fn=HalluDataset.pad)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.eval()
            fwd_func = self.bert_train if "gpt" not in self.args.rep_model else self.gpt_train
            hallu_scores = []
            cnt = 0
            for _, batch in enumerate(unannotated_dataloader):
                input_ids, input_mask, segment_ids, predict_mask, label_ids, guids = batch
                batch = tuple(t.to(self.device) for t in batch[:-1])
                print("id {} mask {} segment {}, pmask {}, label {}".format(input_ids.size(), input_mask.size(),
                                                            segment_ids.size(), predict_mask.size(), label_ids.size()))
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                score = F.softmax(score, dim=1).tolist()
                for sc in score:
                    hallu_scores.append(sc)
            print(len(rep_pos))
            print(len(hallu_scores))
            print(tsen)
            ttokens = tsen.strip().split()
            for pos, hallu in zip(rep_pos, hallu_scores):
                print("{} : {}".format(" ".join(ttokens[pos[0]:pos[1]+1]), hallu[1]))

      def get_model_scores(self, modelpath,
            unannotatedpath="../data_collections/wiki5k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate",
                           epoch=None):
            checkpoint = torch.load(modelpath)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.eval()

            dp = DataProcessor()
            unannotated_examples = dp.get_examples(unannotatedpath, require_uidx=True)
            unannotated_dataset = HalluDataset(unannotated_examples, self.rep_tokenizer, 200, self.rep_name)

            unannotated_dataloader = data.DataLoader(dataset=unannotated_dataset,
                                                       batch_size=8,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       collate_fn=HalluDataset.pad)
            fwd_func = self.bert_train if "gpt" not in self.args.rep_model else self.gpt_train
            pred_dict = {}
            cnt = 0
            for step, batch in enumerate(unannotated_dataloader):
                guids = batch[-1]
                batch = tuple(t.to(self.device) for t in batch[:-1])
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                scores = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                scores = F.softmax(scores, dim=1).tolist()
                for guid, score in zip(guids, scores):
                    pred_dict[int(guid)] = score
                if cnt % 10 == 0:
                    print("Inference - {}".format(cnt*12))
                cnt += 1
            pathname = "../data_preparation/wikinn10k+...data_merge_4424_{}_lr{}_dp{}".format(
                        self.args.name, self.args.lr, self.args.dropout)
            path = pathname if epoch is None else pathname+"."+str(epoch)
            fw = codecs.open(path, "w+", encoding="utf-8")
            # fw = codecs.open("../data_collections/wiki5k+sequential+...+sample_1.data_merge_plus_tianyu_542", "w+", encoding="utf-8")
            # fw = codecs.open("../data_collections/wiki5k+sequential+...+sample_1.data_merge_plus_tianyu_1786", "w+", encoding="utf-8")
            with codecs.open(unannotatedpath, 'r', encoding="utf-8") as fr:
                for entry in fr:
                    example = json.loads(entry.strip())
                    score = pred_dict[example["idx"]]
                    example["consist_model_score"] = score[0]
                    example["hallu_model_score"] = score[1]
                    fw.write(json.dumps(example) + "\n")

      def bert_train(self, input_ids, input_mask, segment_ids, predict_mask):
            model = self.rep_model

            if self.rep_name == "distillbert":
                prediction_scores, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
                # print(hidden_states.size())
            elif "deberta" in self.rep_name:
                hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
            elif "xlnet" in self.rep_name:
                _, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
                hidden_states = [h.transpose(0, 1) for h in hidden_states]
            elif "gpt" in self.rep_name:
                _, _, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
            else:
                prediction_scores, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
                # predictions = model(input_ids=input_ids, attention_mask=input_mask)
                # show_output_size(predictions)
            # print("predict_mask {}".format(predict_mask.size()))
            # for i, h in enumerate(hidden_states):
            #     print("{} : {}".format(i, h.size()))
            '''
            print(hidden_states[0].size())
            # print(predict_mask.size())
            print("{} hidden_states {}".format(len(hidden_states), hidden_states[-1].size()))
            '''
            features = hidden_states if "deberta" in self.rep_name else hidden_states[-1]
            # show_output_size(features)
            state = features * predict_mask.unsqueeze(-1)
            predict_len = predict_mask.sum(1)
            # print("state {}".format(state.size()))
            '''
            print(rep_sen)
            print(rep_subtokens)
            print(rep_subtokens[rep_mask_start_id:rep_mask_end_id])
            print((rep_mask_start_id, rep_mask_end_id))
            '''
            # maxpool_state = torch.max(hidden_states[-1][0][rep_mask_start_id:rep_mask_end_id], dim=0)[0]
            meanpool_state = 1.0 * torch.max(state, dim=1)[0]
            meanpool_state = self.dropout(meanpool_state)
            # print(meanpool_state)
            # print(maxpool_state)
            # print("maxpool_state {}".format(meanpool_state.size()))
            score = self.hidden2label(meanpool_state)
            # print("score {}".format(score.size()))
            return score

      def bert_score(self, input_ids, input_mask, segment_ids, predict_mask):
            model = self.rep_model

            prediction_scores, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
            features = hidden_states if "deberta" in self.rep_name else hidden_states[-1]
            state = features * predict_mask.unsqueeze(-1)
            predict_len = predict_mask.sum(1)
            # print("state {}".format(state.size()))
            '''
            print(rep_sen)
            print(rep_subtokens)
            print(rep_subtokens[rep_mask_start_id:rep_mask_end_id])
            print((rep_mask_start_id, rep_mask_end_id))
            '''
            # maxpool_state = torch.max(hidden_states[-1][0][rep_mask_start_id:rep_mask_end_id], dim=0)[0]
            meanpool_state = 1.0 * torch.max(state, dim=1)[0]
            meanpool_state = self.dropout(meanpool_state)
            # print(meanpool_state)
            # print(maxpool_state)
            # print("maxpool_state {}".format(meanpool_state.size()))
            score = self.hidden2label(meanpool_state)
            # print("score {}".format(score.size()))
            return score

      # def get_word_features(self, data_path="../human_annotation/uhrs_src/data_merge_tag1.txt"):

      def gpt_train(self, src_sen, rep_sen, src_ids, rep_ids):
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
            rets = model(rep_input_ids)
            hidden_states = rets[-1]
            # print("{} hidden_states {}".format(len(hidden_states), hidden_states[-1].size()))

            meanpool_state = torch.mean(hidden_states[self.layer][0][rep_mask_start_id:rep_mask_end_id], dim=0)
            meanpool_state = self.dropout(meanpool_state)

            score = self.hidden2label(meanpool_state)
            # print("score {}".format(score.size()))
            return score.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", default="bert-large-uncased", type=str)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--save_name", default="model", type=str)
    parser.add_argument("--mode", default=1, type=int)
    parser.add_argument("--task_mode", default="offline", type=str)
    parser.add_argument("--clen", default=200, type=int)


    parser.add_argument("--bert_layer", default=10, type=int)
    parser.add_argument("--bert_mask", action="store_true")
    parser.add_argument("--num_epoch", default=20, type=int)
    parser.add_argument("--params", default="map", type=str)

    parser.add_argument("--load_ft_gpt", action="store_true")
    parser.add_argument("--load_model", default="../finetune_learn/output", type=str)
    parser.add_argument("--rep_model", default="bert", type=str)

    parser.add_argument("--context_len", default=1, type=int)
    parser.add_argument("--replace_ratio", default=0.6, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--min_replace_time", default=0, type=int)
    parser.add_argument("--sample_num", default=1, type=int)
    parser.add_argument("--context_limit", default=0, type=int)
    parser.add_argument("--mode_name", default="len-fix", type=str)
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
    # rep_op.svm()
    # rep_op.run(path="selected_replaced_instance.annotate.finish")

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

    # rep_op.get_model_scores("runs/unmask_unbalance_bert_lr_0.0005_dp0.0/model.pt")
    # rep_op.get_model_scores("runs/unmask_downsample_bert_lr_0.0005_dp0.0/model.pt")
    # rep_op.get_model_scores("runs/unmask_enrich_bert_lr_0.0005_dp0.0/model.pt")
    # rep_op = nn.DataParallel(rep_op)

    # rep_op.get_model_scores("runs/datam4424_lr_0.005_dp0.2/model_10_0.5865698729582577_0.8130641720006565.pt",
    #     unannotatedpath="../data_preparation/wikin10k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate")

    if args.test_gpt3:
        sen = "brooks left philadelphia for european waters 26 august 1920 . she was first assigned to the baltic patrol for a short time and then the naval forces in the adriatic sea . she was assigned to the italian fleet in the mediterranean in 1921 and remained there until the end of the year ."
        rep_op.sent_hallu_vis(sen)
    else:
        try:
            rep_op.bert_run(optimizer, epoch=args.num_epoch)
        except KeyboardInterrupt:
            print("Stop by Ctrl-C ...")
        # rep_op.get_model_scores("runs/{}_lr_{}_dp{}/best_model.pt".format(args.name, args.lr, args.dropout))

    '''
    # Balanced dataset by downsampling Hallucination Instances
    rep_op.bert_run(optimizer, trainpath="../human_annotation/uhrs_src/data_merge_plus_tianyu_annotate_balance.txt",
        testpath="../data_collections/selected_replaced_instance.annotate.finish",
        epoch=args.num_epoch)
    '''

    '''
    # Balanced dataset by upsampling Consistent Instances
    rep_op.bert_run(optimizer, trainpath="../human_annotation/uhrs_src/data_merge_plus_tianyu_annotate_enrich_balance.txt",
        testpath="../data_collections/selected_replaced_instance.annotate.finish",
        epoch=args.num_epoch)
    '''
