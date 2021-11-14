import torch
from transformers import BertTokenizer
import xml.etree.ElementTree as ET
import numpy as np
import os
import pickle
import re


class Data(torch.utils.data.Dataset):
    def __init__(self, data_path, saved_data_path, data_idx, label2idx, is_training=True):
        self.is_traing = is_training
        self.document_data = None
        self.data = []
        self.document_max_length = 512
        self.sentence_max_length = 150
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        print('Reading data from {}.'.format(data_path))
        if os.path.exists(saved_data_path):
            with open(file=saved_data_path, mode='rb') as fr:
                info = pickle.load(fr)
                self.document_data = info['data']
            print('load preprocessed data from {}.'.format(saved_data_path))
        else:
            self.document_data = []
            count = 0
            tree = ET.parse(data_path)
            root = tree.getroot()
            for doc in root[0]:
                id = doc.attrib['id']
                label = label2idx[doc.attrib['document_level_value']]
                sentence_list = []
                trigger_word_list = []
                flag = False
                for sent in doc:
                    if sent.text == '-EOP-.' or sent.text == '。':
                        continue

                    s = ''
                    for t in sent.itertext():
                        s += t
                    s = s.replace('-EOP-.', '。').lower()
                    if re.match(r'\d{4}\D\d{2}\D\d{2}\D\d{2}:\d{2}\D$', s) is not None:
                        flag = True
                        continue
                    elif flag:
                        flag = False
                        if len(sent) == 0:
                            continue
                    if len(s)<=4:
                        continue
                    data = self.tokenizer(s, return_tensors='pt', padding='max_length', truncation=True, max_length=150)
                    sent_info = {'trigger_num': 0,
                                 'data': data['input_ids'],
                                 'attention': data['attention_mask']}
                    if len(sent) > 0:
                        # has triggers
                        tmp = sent.text.lower() if sent.text is not None else ''
                        for event in sent:
                            tmp_subwords = self.tokenizer.tokenize(tmp)
                            trigger_subwords = self.tokenizer.tokenize(event.text.lower())
                            pos0 = len(tmp_subwords) + 1
                            pos1 = pos0 + len(trigger_subwords)
                            if pos0 >= self.sentence_max_length - 1:
                                break
                            if pos1 >= self.sentence_max_length - 1:
                                pos1 = self.sentence_max_length - 1
                            else:
                                assert self.tokenizer.convert_ids_to_tokens(data['input_ids'][0, pos0:pos1]) == trigger_subwords

                            trigger_word_idx = torch.zeros(self.sentence_max_length)
                            trigger_word_idx[pos0:pos1] = 1.0 / (pos1 - pos0)

                            trigger_word_list.append({'sent_id': len(sentence_list),
                                                      'idx': trigger_word_idx,
                                                      'value': label2idx[event.attrib['sentence_level_value']]})
                            tmp += event.text.lower()
                            if event.tail is not None:
                                tmp += event.tail.lower()
                            sent_info['trigger_num'] += 1
                    sentence_list.append(sent_info)
                    if len(sentence_list) >= 35:
                        break

                trigger = ''
                for sent in doc:
                    if len(sent) > 0:
                        s = ''
                        for t in sent.itertext():
                            s += t
                        s = s.replace('-EOP-.', '。').lower()
                        trigger += s
                trigger_data = self.tokenizer(trigger, return_tensors='pt', padding='max_length', truncation=True, max_length=260)

                # construct graph
                # 1 document node, 35 sentence nodes, 20 trigger nodes for english data,
                # 1 document node, 35 sentence nodes, 45 trigger nodes for chinese data
                graph = self.create_graph(sentence_list, trigger_word_list)
                assert len(graph)==81

                if len(trigger_word_list)>0:
                    self.document_data.append({
                        'ids': id,
                        'labels': label,
                        'triggers': trigger_data['input_ids'],
                        'trigger_masks': trigger_data['attention_mask'],
                        'sentences': sentence_list,
                        'trigger_words': trigger_word_list,
                        'graphs': graph
                    })
                else:
                    print(id)
                    count+=1

            print('count: ',count)
            # save data
            with open(file=saved_data_path, mode='wb') as fw:
                pickle.dump({'data': self.document_data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(data_path, saved_data_path))

        for i in data_idx:
            self.data.append(self.document_data[i])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_list = self.data[idx]['sentences']
        data_list = []
        attention_list = []
        for s in sentence_list:
            data_list.append(s['data'])
            attention_list.append(s['attention'])
        data = torch.cat(data_list, dim=0)
        attention = torch.cat(attention_list, dim=0)
        sentence_num = len(data)

        trigger_word_list = self.data[idx]['trigger_words']
        sent_idx_list = []
        trigger_word_idx_list = []
        trigger_label_list = []
        for t in trigger_word_list:
            sent_idx_list.append(t['sent_id'])
            trigger_word_idx_list.append(t['idx'])
            trigger_label_list.append(t['value'])
        sent_idx = torch.tensor(sent_idx_list)
        trigger_word_idx = torch.stack(trigger_word_idx_list, dim=0)
        trigger_label = torch.tensor(trigger_label_list)
        graph = torch.tensor(self.data[idx]['graphs']).unsqueeze(0)

        return self.data[idx]['ids'], \
               torch.tensor(self.data[idx]['labels'], dtype=torch.long), \
               self.data[idx]['triggers'], \
               self.data[idx]['trigger_masks'], \
               data, \
               attention, \
               sent_idx, \
               trigger_word_idx, \
               trigger_label, \
               sentence_num, graph


    def create_graph(self, sentence_list, trigger_word_list):
        graph = np.zeros((81, 81))
        sent_num = len(sentence_list)
        trigger_num = len(trigger_word_list)

        # add neighbor edges
        for i in range(1, sent_num):
            graph[i][i+1] = 1.0
            graph[i+1][i] = 1.0

        # add global edges
        for i in range(1, sent_num+1):
            graph[0][i] = 1.0
            graph[i][0] = 1.0

        for i in range(trigger_num):
            j = i+sent_num+1
            graph[0][j] = 1.0
            graph[j][0] = 1.0
            graph[j][trigger_word_list[i]['sent_id'] + 1] = 1.0
            graph[trigger_word_list[i]['sent_id'] + 1][j] = 1.0
        return graph