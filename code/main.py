import torch
import torch.utils.data as D
import torch.nn.functional as F
from dataset import Data
from config import opt
from model import GCN_Joint_EFP
from transformers import AdamW
import torch.nn as nn
import os
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def k_fold_split(data_path):
    train_idx, test_idx = [], []
    labels = []
    label2idx = {}

    tree = ET.parse(data_path)
    root = tree.getroot()
    for document_set in root:
        for document in document_set:
            id = document.attrib['id']
            if id!='ED1397':
                label = document.attrib['document_level_value']
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
                labels.append(label2idx[label])
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for train, test in skf.split(np.zeros(len(labels)), labels):
        train_idx.append(train)
        test_idx.append(test)
    return train_idx, test_idx, label2idx

def collate(samples):
    id, label, trigger, trigger_mask, data, attention, \
    sent_idx, trigger_word_idx, trigger_label, sent_num, graph = map(list, zip(*samples))

    batched_ids = tuple(id)
    batched_labels = torch.tensor(label)
    batched_triggers = torch.cat(trigger, dim=0)
    batched_trigger_mask = torch.cat(trigger_mask, dim=0)
    batched_data = torch.cat(data, dim=0)
    batched_attention = torch.cat(attention, dim=0)
    batched_sent_idx = sent_idx
    batched_trigger_word_idx = trigger_word_idx
    batched_trigger_labels = torch.cat(trigger_label, dim=0)
    batched_sent_num = torch.tensor(sent_num)
    batched_graph = torch.cat(graph, dim=0)
    return batched_ids, batched_labels, batched_triggers, batched_trigger_mask, \
           batched_data, batched_attention, \
           batched_sent_idx, batched_trigger_word_idx, batched_trigger_labels, batched_sent_num, \
           batched_graph


def get_data(train_idx, test_idx, label2idx):
    trainset = Data(opt.data_path, opt.saved_path, train_idx, label2idx, is_training=True)
    train_loader = D.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=10, collate_fn=collate)
    testset = Data(opt.data_path, opt.saved_path, test_idx, label2idx, is_training=False)
    test_loader = D.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=10, collate_fn=collate)
    return train_loader, test_loader


def train(model, trainloader, optimizer, opt):
    model.train()
    # start_time = time.time()

    loss_list = []
    for batch_idx, (ids, labels, triggers, trigger_masks, words, masks, sent_idx, trigger_word_idx, trigger_labels,
                    sent_nums, graphs) in enumerate(trainloader):
        if opt.gpu:
            triggers = triggers.cuda()
            trigger_masks = trigger_masks.cuda()
            words = words.cuda()
            masks = masks.cuda()
            # trigger_word_idx = trigger_word_idx.cuda()
            graphs = graphs.cuda()
            labels = labels.cuda()
            sent_nums = sent_nums.cuda()
            trigger_labels = trigger_labels.cuda()
        sent_idx_list = []
        trigger_word_idx_list = []
        for i in range(len(sent_idx)):
            sent_idx_list.append(sent_idx[i].cuda())
            trigger_word_idx_list.append(trigger_word_idx[i].cuda())
        optimizer.zero_grad()
        logit, trigger_logit, kl_loss = model(ids=ids, triggers=triggers, trigger_masks=trigger_masks, words=words, masks=masks,
                                    sent_idx=sent_idx_list, trigger_word_idx=trigger_word_idx_list, sent_nums=sent_nums, graphs=graphs)
        main_loss = nn.functional.cross_entropy(logit, labels)
        aux_loss = nn.functional.cross_entropy(trigger_logit, trigger_labels)
        loss = main_loss + kl_loss

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    # print("time:%.3f" % (time.time() - start_time))
    return np.mean(loss_list)


def evaluate(model, test_loader, filepath=None):
    if filepath is not None:
        f = open(filepath, 'w')
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (ids, labels, triggers, trigger_masks, words, masks, sent_idx, trigger_word_idx, trigger_labels,
                        sent_nums, graphs) in enumerate(test_loader):
            if opt.gpu:
                triggers = triggers.cuda()
                trigger_masks = trigger_masks.cuda()
                words = words.cuda()
                masks = masks.cuda()
                # sent_idx = sent_idx.cuda()
                # trigger_word_idx = trigger_word_idx.cuda()
                graphs = graphs.cuda()
                labels = labels.cuda()
                sent_nums = sent_nums.cuda()
                trigger_labels = trigger_labels.cuda()
            sent_idx_list = []
            trigger_word_idx_list = []
            for i in range(len(sent_idx)):
                sent_idx_list.append(sent_idx[i].cuda())
                trigger_word_idx_list.append(trigger_word_idx[i].cuda())
            logit, _, _ = model(ids=ids, triggers=triggers, trigger_masks=trigger_masks, words=words, masks=masks,
                             sent_idx=sent_idx_list, trigger_word_idx=trigger_word_idx_list, sent_nums=sent_nums, graphs=graphs)
            _, predicted = torch.max(logit.data,1)
            correct += predicted.data.eq(labels.data).cpu().sum()
            y_true += labels.cpu().data.numpy().tolist()
            y_pred += predicted.cpu().data.numpy().tolist()

            if filepath is not None:
                batch = labels.shape[0]
                for i in range(batch):
                    f.write(ids[i] + "\t" + str(labels[i].item()) + "\t" + str(predicted[i].item()) + "\n")

    f1_micro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='micro')
    f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
    if filepath is not None:
        f.write("f1_micro: " + str(f1_micro) + "\n")
        f.write("f1_macro: " + str(f1_macro) + "\n")
        f.close()
    return f1_micro, f1_macro



if __name__=='__main__':
    train_idx, test_idx, label2idx = k_fold_split(opt.data_path)
    f1_micro_list = []
    f1_macro_list = []
    for i in range(10):
        model_path = opt.model_path + "_" + str(i) + ".pt"
        output_path = opt.output_path + "_" + str(i) + ".txt"
        train_loader, test_loader = get_data(train_idx[i], test_idx[i], label2idx)
        model = GCN_Joint_EFP(opt, len(label2idx))
        if opt.gpu:
            model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=opt.lr)
        max_f1 = 0
        max_f1_micro = 0
        max_f1_macro = 0
        for epoch in range(opt.n_epochs):
            train_loss = train(model, train_loader, optimizer, opt)
            test_f1_micro, test_f1_macro = evaluate(model, test_loader)
            print("Epoch:%d-%d loss:%f F1_micro:%.2f F1_macro:%.2f" % (
            i, epoch, train_loss, test_f1_micro * 100, test_f1_macro * 100))
            if test_f1_micro + test_f1_macro > max_f1:
                max_f1 = test_f1_micro + test_f1_macro
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_f1_micro, test_f1_macro = evaluate(model, test_loader, filepath=output_path)
        print("Epoch:%d-%d F1_micro:%.2f F1_macro:%.2f" % (
        i, checkpoint['epoch'], test_f1_micro * 100, test_f1_macro * 100))
        f1_micro_list.append(test_f1_micro)
        f1_macro_list.append(test_f1_macro)

    output = open(opt.output_path + ".txt", "w")
    f1_micro_a = np.mean(f1_micro_list)
    f1_macro_a = np.mean(f1_macro_list)
    output.write("batch_size=" + str(opt.batch_size) + "\n")
    output.write("lr=" + str(opt.lr) + "\n")
    output.write("f1_micro_a: " + str(f1_micro_a) + "\n")
    output.write("f1_macro_a: " + str(f1_macro_a) + "\n")
    print("F1_micro_a: %.2f F1_macro_a: %.2f" % (f1_micro_a * 100, f1_macro_a * 100))

    ct_p = []
    ct_m = []
    ps_p = []
    for i in range(10):
        filename = opt.output_path + "_" + str(i) + ".txt"
        y_true = []
        y_pred = []
        with open(filename, "r") as f:
            for l in f.readlines():
                line = l.split()
                if len(line) < 3:
                    break
                y_true.append(line[1])
                y_pred.append(line[2])

        with open(filename, "a") as f:
            t_ct_p = f1_score(y_true, y_pred, labels=[0], average="macro")
            f.write("CT+: " + str(t_ct_p) + "\n")
            ct_p.append(t_ct_p)

            t_ct_m = f1_score(y_true, y_pred, labels=[1], average="macro")
            f.write("CT-: " + str(t_ct_m) + "\n")
            ct_m.append(t_ct_m)

            t_ps_p = f1_score(y_true, y_pred, labels=[2], average="macro")
            f.write("PS+: " + str(t_ps_p) + "\n")
            ps_p.append(t_ps_p)

    output.write("CT+: " + str(np.mean(ct_p)) + "\n")
    output.write("CT-: " + str(np.mean(ct_m)) + "\n")
    output.write("PS+: " + str(np.mean(ps_p)) + "\n")
    output.close()



