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

            self.load_model = args.load_model

            if "xlnet" in args.rep_model:
                self.tokenizer = AutoTokenizer.from_pretrained(self.load_model)
                self.model = XLNetLMHeadModel.from_pretrained(self.load_model, mem_len=1024).to(args.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.load_model)
                config = AutoConfig.from_pretrained(self.load_model)
                config.output_hidden_states = True
                self.model = AutoModelWithLMHead.from_pretrained(self.load_model, config=config).to(args.device)


            hidden_size = 1024 if "large" in self.load_model or self.load_model=="gpt2-medium" else 768

            self.hidden2label = nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_size//2, 2)).to(args.device)
            self.dropout = torch.nn.Dropout(args.dropout)
            self.layer = args.bert_layer

            self.eval()
            self.device = args.device
            self.args = args


      def bert_run(self, optim, trainpath="../data_collections/Wiki-Hades/train.txt",
                   testpath="../data_collections/Wiki-Hades/test.txt",
                   validpath="../data_collections/Wiki-Hades/valid.txt",
                   epoch=10):
              # testpath="selected_replaced_instance.annotate.finish", epoch=10):
            prefix = "runs/{}_lr_{}_dp_{}_{}_clen{}/".format(self.load_model, self.args.lr,
                                            self.args.dropout, self.args.task_mode, self.args.clen)
            bestmodelpath = prefix + "best_model.pt"
            epoch_start = 1
            if os.path.exists(bestmodelpath) and self.args.continue_train:
                checkpoint = torch.load(bestmodelpath)
                self.load_state_dict(checkpoint["model_state_dict"])
                epoch_start = checkpoint["epoch"] + 1

            writer = SummaryWriter(prefix)
            csvlogger = prefix + "valid_log.csv"

            if not os.path.exists(csvlogger):
                csvfile = open(csvlogger, 'w+')
                fileHeader = ["epoch", "H_p", "H_r", "H_f1", "C_p", "C_r", "C_f1", "Gmean",
                              "Acc", "BSS", "ROC_AUC"]
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fileHeader)
            else:
                csvfile = open(csvlogger, 'a')
                csvwriter = csv.writer(csvfile)

            dp = DataProcessor()
            train_examples = dp.get_examples(trainpath)

            train_dataset = HalluDataset(train_examples, self.tokenizer, self.args.clen,
                                         self.load_model, self.args.task_mode)

            train_dataloader = data.DataLoader(dataset=train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)
            nSamples = dp.get_label_dist()
            nTrain = sum(nSamples)
            print("====Train label : {}".format(nSamples))
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]

            '''
            test_examples = dp.get_examples(testpath)
            test_dataset = HalluDataset(test_examples, self.tokenizer, self.args.clen,
                                        self.load_model, self.args.task_mode)
            test_dataloader = data.DataLoader(dataset=test_dataset,
                                               batch_size=4,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)
            '''
            valid_examples = dp.get_examples(validpath)
            valid_dataset = HalluDataset(valid_examples, self.tokenizer, self.args.clen,
                                         self.load_model, self.args.task_mode)
            valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                               batch_size=4,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)

            normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
            loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)

            fwd_func = self.bert_train
            best_acc, best_f1_score = -1, -1
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
                    train_loss += loss.item()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    cnt += 1
                    if cnt % 10 == 0:
                        print("Training Epoch {} - {}% - {}".format(ei, int(100 * cnt/nTrain), train_loss/cnt))
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


      def bert_train(self, input_ids, input_mask, segment_ids, predict_mask):
            model = self.rep_model

            if self.load_model == "distillbert":
                prediction_scores, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
                # print(hidden_states.size())
            elif "deberta" in self.load_model:
                hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
            elif "xlnet" in self.load_model:
                _, hidden_states = model(input_ids=input_ids, attention_mask=input_mask)
                hidden_states = [h.transpose(0, 1) for h in hidden_states]
            elif "gpt" in self.load_model:
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
            features = hidden_states if "deberta" in self.load_model else hidden_states[-1]
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
            features = hidden_states if "deberta" in self.load_model else hidden_states[-1]
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
    parser.add_argument("--finetune_from", default="../finetune_learn/output", type=str)
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
