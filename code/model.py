import torch.nn as nn
from transformers import BertModel, BertConfig
import torch
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)


class UGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(UGCN, self).__init__()
        self.gcn_layer1 = GCNLayer(in_dim, hid_dim)
        self.gcn_layer2 = GCNLayer(hid_dim, out_dim)
        self.gcn_layer1_var = GCNLayer(in_dim, hid_dim)
        self.gcn_layer2_var = GCNLayer(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.trans_mean = nn.Linear(in_dim, in_dim)
        self.trans_var = nn.Linear(in_dim, in_dim)

    def forward(self, adj, gcn_inputs):
        output_features = []
        out_vars = []
        output_features.append(gcn_inputs)
        out_vars.append(gcn_inputs)
        adj_list = []
        adj_var_list = []
        for i in range(adj.size()[0]):
            adj_list.append(self.normalize(adj[i].view(adj.size()[1], adj.size()[2])))
            adj_var_list.append(self.normalize1(adj[i].view(adj.size()[1], adj.size()[2])))
        adj = torch.cat(adj_list, dim=0)
        adj_var = torch.cat(adj_var_list, dim=0)
        adj = adj.type_as(gcn_inputs)
        adj_var = adj_var.type_as(gcn_inputs)
        mean_vectors = F.relu(self.trans_mean(gcn_inputs))
        var_vectors = F.relu(self.trans_var(gcn_inputs))
        output_features.append(mean_vectors)
        out_vars.append(var_vectors)

        node_weight = torch.exp(-0.001*var_vectors)
        x_mean = mean_vectors.mul(node_weight)
        x_var = var_vectors.mul(node_weight).mul(node_weight)
        Ax_mean = adj.bmm(x_mean)
        Ax_var = adj_var.bmm(x_var)
        hid_output_mean = self.gcn_layer1(Ax_mean)
        hid_output_var = self.gcn_layer1_var(Ax_var)
        output_features.append(hid_output_mean)
        out_vars.append(hid_output_var)

        node_weight = torch.exp(-0.001*hid_output_var)
        x_mean = hid_output_mean.mul(node_weight)
        x_var = hid_output_var.mul(node_weight).mul(node_weight)
        Ax_mean = adj.bmm(x_mean)
        Ax_var = adj_var.bmm(x_var)
        output_mean = self.gcn_layer2(Ax_mean)
        output_var = self.gcn_layer2_var(Ax_var)
        sample_v = torch.randn(1, 1)[0][0]
        output_mean = output_mean + (torch.sqrt(output_var+1e-8)*sample_v)
        output_features.append(output_mean)
        out_vars.append(output_var)
        return output_features, out_vars

    def normalize(self, A, symmetric=True):
        # A = A+I
        A = A + torch.eye(A.size(0)).cuda()
        # degree of nodes
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D).unsqueeze(0)

    def normalize1(self, A, symmetric=True):
        # A = A+I
        A = A + torch.eye(A.size(0)).cuda()
        # degree of nodes
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A).mm(D).unsqueeze(0)


class GCN_Joint_EFP(nn.Module):
    def __init__(self, config, y_num):
        super(GCN_Joint_EFP, self).__init__()
        self.config = config
        self.y_num = y_num
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.bert = BertModel.from_pretrained('bert-base-chinese')   # bert-base-uncased for english data, bert-base-chinese for chinese data
        print('The number of parameters of bert: ',
              sum(p.numel() for p in self.bert.parameters() if p.requires_grad))

        self.gcn_in_dim = config.bert_hid_size
        self.gcn_hid_dim = config.gcn_hid_dim
        self.gcn_out_dim = config.gcn_out_dim
        self.dropout = config.dropout

        self.ugcn = UGCN(self.gcn_in_dim, self.gcn_hid_dim, self.gcn_out_dim, self.dropout)

        self.bank_size = self.gcn_in_dim + self.gcn_hid_dim + self.gcn_out_dim
        self.linear_dim = config.linear_dim
        self.predict = nn.Linear(self.bank_size, self.y_num)
        # self.trigger_predict = nn.Linear(self.bank_size, self.y_num)


    def forward(self, **params):
        triggers = params['triggers']
        trigger_masks = params['trigger_masks']
        bsz = triggers.size()[0]
        doc_outputs = self.bert(triggers, attention_mask=trigger_masks)
        document_cls = doc_outputs[1]

        words = params['words']  # [bsz, seq_len]
        masks = params['masks']  # [bsz, seq_len]
        sent_outputs = self.bert(words, attention_mask=masks)  # sentence_cls: [bsz, bert_dim]
        sentence_embed = sent_outputs[0]
        sentence_cls = sent_outputs[1]

        sent_idx = params['sent_idx']  # bsz * [trigger_num]
        trigger_word_idx = params['trigger_word_idx']  # bsz * [trigger_num, seq_len]
        graphs = params['graphs']
        assert graphs.size()[0] == bsz, "batch size inconsistent"

        split_sizes = params['sent_nums'].tolist()
        # for i in range(bsz):
        #     sentence_num = graphs[i].number_of_nodes('node') - 1 - sent_idx[i].shape[0]
        #     split_sizes.append(sentence_num)
        feature_list = list(torch.split(sentence_cls, split_sizes, dim=0))  # bsz * [num, bert_dim]
        sentence_embed_list = list(torch.split(sentence_embed, split_sizes, dim=0))

        sentence_trigger = []
        trigger_nums = []
        for i in range(bsz):
            # extracr sentences containing triggers
            t = sentence_embed_list[i].index_select(0, sent_idx[i])  # [trigger_num, seq_len, bert_dim]
            # extract trigger embeds
            trigger_embed = torch.sum(trigger_word_idx[i].unsqueeze(-1) * t, dim=1)   # [trigger_num, bert_dim]
            # assert trigger_embed.size()[0]==sent_idx[i].size()[-1]
            trigger_nums.append(trigger_embed.size()[0])
            fea = torch.cat((feature_list[i], trigger_embed), dim=0)
            pad = torch.zeros(graphs.size()[1]-1-fea.size()[0], fea.size()[-1]).cuda()
            fea = torch.cat((fea, pad), dim=0).unsqueeze(0)
            assert fea.size()[1]==graphs.size()[1]-1
            sentence_trigger.append(fea)
        sentence_trigger = torch.cat(sentence_trigger, dim=0)
        features = torch.cat((document_cls.unsqueeze(1), sentence_trigger), dim=1)
        assert features.size()[0]==bsz
        assert features.size()[1]==graphs.size()[1]

        output_features, output_means = self.ugcn(graphs, features)   # [bsz, num_node, dim]
        output_feature_list = [output_features[0], output_features[2], output_features[3]]
        output_feature = torch.cat(output_feature_list, dim=-1)
        document_features = []
        trigger_features = []
        for i in range(bsz):
            document_features.append(output_feature[i:i+1, 0, :])
            trigger_start = 1 + split_sizes[i]
            trigger_end = trigger_start+trigger_nums[i]
            trigger_features.append(output_feature[i:i+1, trigger_start:trigger_end, :].view(-1, output_feature.size()[-1]))

        document_feature = torch.cat(document_features, dim=0).view(-1, output_feature.size()[-1])
        trigger_feature = torch.cat(trigger_features, dim=0).view(-1, output_feature.size()[-1])

        # classification
        predictions = self.predict(document_feature)
        # trigger_predictions = self.trigger_predict(trigger_feature)
        trigger_predictions = trigger_feature

        mean = output_features[1]   # (bsz, node_num, gcn_out_dim)
        var = output_means[1]
        KL_divergence = 0.5*torch.mean(torch.square(mean)+var-torch.log(1e-8+var)-1, dim=-1)
        KL_divergence = torch.mean(KL_divergence)
        KL_loss = 5e-4*KL_divergence

        return predictions, trigger_predictions, KL_loss
