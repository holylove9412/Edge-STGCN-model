import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FeatureFusionLayer(nn.Module):
    def __init__(self, input_dim, node_size, fusion_method='dense'):
        super(FeatureFusionLayer, self).__init__()
        self.fusion_method = fusion_method
        self.node_size = node_size
        if fusion_method == 'conv1D':
            self.conv = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1)
        elif fusion_method == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            )
        elif fusion_method == 'dense':
            self.line = nn.Linear(2 * input_dim, input_dim)
        else:
            self.nodevec1 = nn.Parameter(torch.randn(input_dim, input_dim // 2))
            self.bias1 = nn.Parameter(torch.zeros(input_dim // 2))
            self.nodevec2 = nn.Parameter(torch.randn(input_dim, input_dim // 2))
            self.bias2 = nn.Parameter(torch.zeros(input_dim // 2))

    def forward(self, inputs):
        if self.fusion_method == 'conv1D':
            if len(inputs) == 3:
                x1, x2, x3 = inputs
                fused1 = F.relu(self.conv(x1.permute(0, 2, 1))).permute(0, 2, 1)
                fused2 = F.relu(self.conv(x2.permute(0, 2, 1))).permute(0, 2, 1)
                fused3 = F.relu(self.conv(x3.permute(0, 2, 1))).permute(0, 2, 1)
                return torch.cat([fused1, fused2, fused3], dim=-1)
            elif len(inputs) == 2:
                x1, x2 = inputs
                fused1 = F.relu(self.conv(x1.permute(0, 2, 1))).permute(0, 2, 1)
                fused2 = F.relu(self.conv(x2.permute(0, 2, 1))).permute(0, 2, 1)
                return torch.cat([fused1, fused2], dim=-1)
        else:
            x1, x2 = inputs
            if self.fusion_method == 'mlp':
                concatenated = torch.cat([x1, x2], dim=-1)
                return self.mlp(concatenated)
            elif self.fusion_method == 'dense':
                concatenated = torch.cat([x1, x2], dim=-1)
                return F.relu(self.line(concatenated))
            else:
                transformed1 = torch.matmul(x1, self.nodevec1) + self.bias1
                transformed2 = torch.matmul(x2, self.nodevec2) + self.bias2
                return torch.cat([transformed1, transformed2], dim=-1)


class EdgeAttributeEmbedding(nn.Module):
    def __init__(self, metrix):
        super(EdgeAttributeEmbedding, self).__init__()

        self.metrix = metrix.float()
        self.w = nn.Parameter(torch.randn_like(self.metrix))
        self.b = nn.Parameter(torch.zeros_like(self.metrix))

    def forward(self, inputs):

        mat = self.w * self.metrix + self.b
        return torch.matmul(mat, inputs)

#EdgeSTGCN model
class EdgeSTGCN(nn.Module):
    def __init__(self, args):
        super(EdgeSTGCN, self).__init__()
        self.n_node, self.n_in = getattr(args, 'state_shape', (376, 1))
        self.b_in = 1
        self.n_out = 1
        self.seq_in = getattr(args, 'seq_in', 20)
        self.seq_out = getattr(args, 'seq_out', 20)
        self.embed_size = getattr(args, 'embed_size', 64)
        self.hidden_dim = getattr(args, "hidden_dim", 64)
        self.n_sp_layer = getattr(args, "n_sp_layer", 3)
        self.n_tp_layer = getattr(args, "n_tp_layer", 2)
        self.dropout_rate = getattr(args, 'dropout', 0.0)
        self.featureLayer = getattr(args, "featureLayer", 'dense')

        self.adj = torch.tensor(getattr(args, "adj", np.eye(self.n_node)), dtype=torch.float32)
        self.dadj = torch.tensor(getattr(args, "dadj", np.eye(self.n_node)), dtype=torch.float32)
        self.length_adj = torch.tensor(getattr(args, "length_adj", np.eye(self.n_node)), dtype=torch.float32)
        self.invert_adj = torch.tensor(getattr(args, "invert_adj", np.eye(self.n_node)), dtype=torch.float32)
        self.node_edge = torch.tensor(getattr(args, "node_edge"), dtype=torch.float32)
        self.n_edge, self.e_in = getattr(args, 'edge_state_shape', (362, 1))

        self.edge_embed_dadj = EdgeAttributeEmbedding(self.dadj)
        self.edge_embed_inv = EdgeAttributeEmbedding(torch.abs(self.invert_adj).t())
        self.edge_embed_length = EdgeAttributeEmbedding(self.length_adj)

        self.fc_embed_in = nn.Linear(self.n_node * self.n_in, self.embed_size)
        self.fc_embed_proc = nn.Linear(self.embed_size, self.embed_size)


        self.conv1d = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernel_size=3, padding=1)
        self.bn1d = nn.BatchNorm1d(self.embed_size)
        self.pool1d = nn.MaxPool1d(kernel_size=1, stride=1)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.fc_bound = nn.Linear(self.n_node * self.b_in, self.embed_size // 2)

        self.gcn1 = GCNConv(self.embed_size, self.embed_size)
        self.gcn2 = GCNConv(self.embed_size, self.embed_size)
        self.gcn3 = GCNConv(self.embed_size, self.embed_size)

        self.feature_fusion = FeatureFusionLayer(self.embed_size, self.n_node, fusion_method=self.featureLayer)
        self.gru_layers1 = nn.ModuleList([
            nn.GRU(self.embed_size, self.hidden_dim, batch_first=True)
            for _ in range(self.n_tp_layer)
        ])
        self.gru_layers2 = nn.ModuleList([
            nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            for _ in range(self.n_tp_layer)
        ])
        self.fc_out_embed = nn.Linear(self.hidden_dim, self.embed_size)
        self.fc_out = nn.Linear(self.embed_size, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_in, B_in, Adj_in, Node_Edge):
        batch_size = X_in.size(0)
        x = X_in.view(batch_size, self.seq_in, -1)
        x = self.fc_embed_in(x)
        x = self.dropout(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.bn1d(x_conv)
        x_conv = self.pool1d(x_conv)
        x_conv = self.dropout(x_conv)
        res = x_conv[:, :, -1:]
        x_conv = F.relu(x_conv)
        x_proc = x_conv.transpose(1, 2)

        for _ in range(self.n_sp_layer):
            templ_x = F.relu(self.fc_embed_proc(x_proc))
            emb1 = self.edge_embed_dadj(templ_x)
            x1 = F.relu(self.gcn1(emb1, Adj_in))
            x1 = self.dropout(x1)

            emb2 = self.edge_embed_inv(templ_x)
            xx = torch.matmul(Node_Edge, emb2)
            x2 = F.relu(self.gcn2(xx, Adj_in))

            emb3 = self.edge_embed_length(templ_x)
            x3 = F.relu(self.gcn3(emb3, Adj_in))
            x3 = self.dropout(x3)

            x_proc = self.feature_fusion([x1, x2, x3])

            for gru in self.gru_layers1:
                x_proc, _ = gru(x_proc)

            x_proc = x_proc[:, -self.seq_out:, :]

            b = B_in.view(batch_size, self.seq_out, -1)
            x_proc = torch.cat([x_proc, b], dim=-1)

            for _ in range(self.n_sp_layer):
                templ_x = F.relu(self.fc_embed_proc(x_proc))
                emb1 = self.edge_embed_dadj(templ_x)
                x1 = F.relu(self.gcn1(emb1, Adj_in))
                x1 = self.dropout(x1)

                emb2 = self.edge_embed_inv(templ_x)
                xx = torch.matmul(Node_Edge, emb2)
                x2 = F.relu(self.gcn2(xx, Adj_in))

                emb3 = self.edge_embed_length(templ_x)
                x3 = F.relu(self.gcn3(emb3, Adj_in))
                x3 = self.dropout(x3)

                x_proc = self.feature_fusion([x1, x2, x3])
            for gru in self.gru_layers2:
                x_proc, _ = gru(x_proc)
            x_out = self.fc_out_embed(x_proc)
            x_out = self.dropout(x_out)
            res_adjusted = res.repeat(1, self.seq_out, 1)  # (batch_size, seq_out, embed_size)
            x_final = x_out + res_adjusted
            x_final = F.relu(x_final)
        out = self.sigmoid(self.fc_out(x_final))
        out = out.view(batch_size, self.seq_out, self.n_node, self.n_out)
        return out

# STGCN model
class STGCN(nn.Module):
    def __init__(self, args):
        super(STGCN, self).__init__()
        self.n_node, self.n_in = getattr(args, 'state_shape', (376, 1))
        self.b_in = 1
        self.n_out = 1
        self.seq_in = getattr(args, 'seq_in', 20)
        self.seq_out = getattr(args, 'seq_out', 20)
        self.embed_size = getattr(args, 'embed_size', 64)
        self.hidden_dim = getattr(args, "hidden_dim", 64)
        self.n_sp_layer = getattr(args, "n_sp_layer", 3)
        self.n_tp_layer = getattr(args, "n_tp_layer", 2)
        self.dropout_rate = getattr(args, 'dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.fc_embed = nn.Linear(self.n_node * self.n_in, self.embed_size)

        self.conv1d = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernel_size=3, padding=1)
        self.bn1d = nn.BatchNorm1d(self.embed_size)
        self.pool1d = nn.MaxPool1d(kernel_size=1, stride=1)

        self.fc_bound = nn.Linear(self.n_node * self.b_in, self.embed_size // 2)

        self.gcn_layers = nn.ModuleList([GCNConv(self.embed_size, self.embed_size)
                                         for _ in range(self.n_sp_layer)])
        self.gru_layers1 = nn.ModuleList([
            nn.GRU(self.embed_size, self.hidden_dim, batch_first=True)
            for _ in range(self.n_tp_layer)
        ])

        self.fc_concat = nn.Linear(self.hidden_dim + self.embed_size // 2, self.embed_size)

        self.gcn_layers2 = nn.ModuleList([GCNConv(self.embed_size, self.embed_size)
                                          for _ in range(self.n_sp_layer)])

        self.gru_layers2 = nn.ModuleList([
            nn.GRU(self.embed_size, self.hidden_dim, batch_first=True)
            for _ in range(self.n_tp_layer)
        ])
        self.fc_out_embed = nn.Linear(self.hidden_dim, self.embed_size)
        self.fc_out = nn.Linear(self.embed_size, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_in, B_in, Adj_in):
        batch_size = X_in.size(0)
        x = X_in.view(batch_size, self.seq_in, -1)
        x = self.fc_embed(x)
        x = self.dropout(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.bn1d(x_conv)
        x_conv = self.pool1d(x_conv)
        x_conv = self.dropout(x_conv)

        res = x_conv[:, :, -1:]
        x_conv = F.relu(x_conv)

        x = x_conv.transpose(1, 2)

        b = B_in.view(batch_size, self.seq_out, -1)
        b = self.fc_bound(b)
        b = self.dropout(b)

        for gcn in self.gcn_layers:
            x = F.relu(gcn(x, Adj_in))
            x = self.dropout(x)
        for gru in self.gru_layers1:
            x, _ = gru(x)
        x = x[:, -self.seq_out:, :]

        x = torch.cat([x, b], dim=-1)

        x = F.relu(self.fc_concat(x))

        for gcn in self.gcn_layers2:
            x = F.relu(gcn(x, Adj_in))
            x = self.dropout(x)
        for gru in self.gru_layers2:
            x, _ = gru(x)
        x_out = self.fc_out_embed(x)
        x_out = self.dropout(x_out)
        x_final = x_out + res.repeat(1, self.seq_out, 1)
        x_final = F.relu(x_final)
        out = self.sigmoid(self.fc_out(x_final))
        out = out.view(batch_size, self.seq_out, self.n_node, self.n_out)
        return out


# MLP model

class MLPNetwork(nn.Module):
    def __init__(self, args):
        super(MLPNetwork, self).__init__()
        self.n_node, self.n_in = getattr(args, 'state_shape', (376, 1))
        self.b_in = 1
        self.n_out = 1
        self.seq_in = getattr(args, 'seq_in', 20)
        self.seq_out = getattr(args, 'seq_out', 20)
        self.embed_size = getattr(args, 'embed_size', 64)
        self.hidden_dim = getattr(args, "hidden_dim", 64)
        self.n_sp_layer = getattr(args, "n_sp_layer", 3)
        self.n_tp_layer = getattr(args, "n_tp_layer", 2)
        self.dropout_rate = getattr(args, 'dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.fc_embed = nn.Linear(self.n_node * self.n_in, self.embed_size)

        self.conv1d = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernel_size=3, padding=1)
        self.bn1d = nn.BatchNorm1d(self.embed_size)
        self.pool1d = nn.MaxPool1d(kernel_size=1, stride=1)

        self.fc_bound = nn.Linear(self.n_node * self.b_in, self.embed_size // 2)
        #使用全连接层代替空间特征提取层，以保证三个模型架构一致
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.embed_size, self.embed_size)
            for _ in range(self.n_sp_layer)
        ])
        self.gru_layers1 = nn.ModuleList([
            nn.GRU(self.embed_size, self.hidden_dim, batch_first=True)
            for _ in range(self.n_tp_layer)
        ])
        self.gru_layers2 = nn.ModuleList([
            nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            for _ in range(self.n_tp_layer)
        ])
        self.fc_out_embed = nn.Linear(self.hidden_dim, self.embed_size)
        self.fc_out = nn.Linear(self.embed_size, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_in, B_in):
        batch_size = X_in.size(0)
        x = X_in.view(batch_size, self.seq_in, -1)
        x = self.fc_embed(x)
        x = self.dropout(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.bn1d(x_conv)
        x_conv = self.pool1d(x_conv)
        x_conv = self.dropout(x_conv)

        res = x_conv[:, :, -1:]
        x_conv = F.relu(x_conv)

        x = x_conv.transpose(1, 2)

        b = B_in.view(batch_size, self.seq_out, -1)
        b = self.fc_bound(b)
        b = self.dropout(b)
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        for gru in self.gru_layers1:
            x, _ = gru(x)
        x = x[:, -self.seq_out:, :]

        x = torch.cat([x, b], dim=-1)
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)
        for gru in self.gru_layers2:
            x, _ = gru(x)
        x_out = self.fc_out_embed(x)
        x_out = self.dropout(x_out)
        x_final = x_out + res.repeat(1, self.seq_out, 1)
        x_final = F.relu(x_final)
        out = self.sigmoid(self.fc_out(x_final))
        out = out.view(batch_size, self.seq_out, self.n_node, self.n_out)
        return out



class ModelLayer(nn.Module):
    def __init__(self, args, model_type='edge'):
        super(ModelLayer, self).__init__()
        if model_type == 'edge':
            self.model = EdgeSTGCN(args)
        elif model_type == 'stgcn':
            self.model = STGCN(args)
        elif model_type == 'mlp':
            self.model = MLPNetwork(args)
        else:
            raise ValueError("model_type 必须在 'edge', 'stgcn', 'mlp' 中选择")

    def forward(self, *inputs):
        return self.model(*inputs)
