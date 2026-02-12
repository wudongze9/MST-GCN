

import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn.pytorch import RelGraphConv


class MST_RGCN_Layer(nn.Module):
    """
    Multi-Scale Spatio-Temporal Relational Graph Convolutional Layer.
    该层通过一个自适应门控机制，融合短期(t-1)和长期(t-2)的时间依赖，
    并将其与当前的空间信息进行更新。
    """
    def __init__(self, in_feats, out_feats, num_rels, dropout=0.2, self_loop=True):
        super(MST_RGCN_Layer, self).__init__()
        
        # 空间聚合层
        self.rgcn = RelGraphConv(in_feats, out_feats, num_rels=num_rels, self_loop=self_loop)
        
        # 时间更新单元
        self.gru = nn.GRUCell(input_size=out_feats, hidden_size=out_feats)
        
        # --- 核心创新：自适应历史融合门 ---
        self.gate_linear = nn.Linear(out_feats * 3, out_feats)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features, prev_h_t1, prev_h_t2):
        """
        :param g: 当前时间点的 DGLGraph 对象
        :param features: 当前时间点的输入节点特征
        :param prev_h_t1: 来自 t-1 时刻的隐藏状态 (短期记忆)
        :param prev_h_t2: 来自 t-2 时刻的隐藏状态 (长期记忆)
        :return: (h_t, gate) 
        """
        # 1. 空间聚合: 获取当前时刻的空间上下文信息
        etype = g.edata.get('etype', torch.zeros(g.num_edges(), dtype=torch.long, device=g.device))
        spatial_msg = self.rgcn(g, features, etype)
        
        # <--- MODIFIED: 初始化 gate 变量
        gate = None
        
        # --- 2. 多尺度时间融合 ---
        if prev_h_t2 is None:
            fused_h = prev_h_t1
            # <--- MODIFIED: 在 t < 2 时，门控不起作用，可以返回 None 或 0.5 的张量
            #      为了安全地进行 cat 操作，我们返回 None
            gate = None
        else:
            # 门控的决策依据是当前的空间信息和两种历史信息
            gate_input = torch.cat([spatial_msg, prev_h_t1, prev_h_t2], dim=1)
            gate = torch.sigmoid(self.gate_linear(gate_input))
            
            # 门控融合
            fused_h = gate * prev_h_t1 + (1 - gate) * prev_h_t2

        # 3. 时间更新
        h_t = self.gru(spatial_msg, fused_h)
        
        # 4. 应用激活函数和 Dropout
        h_t = self.activation(h_t)
        h_t = self.dropout(h_t)
        
        # <--- MODIFIED: 返回 h_t 和 gate
        return h_t, gate


class Multi_MST_GCN(nn.Module):
    """
    Multi-Scale Spatio-Temporal Relational Graph Convolutional Network.
    """
    def __init__(self, num_layers, in_feats, h_feats, bi, dropout, batch_size, device, course_feat_dim=None):
        super(Multi_MST_GCN, self).__init__()
        self.num_layers = num_layers
        self.h_feats = h_feats
        self.dropout = dropout
        self.device = device

        # 构建多尺度时空GNN层
        self.st_layers = nn.ModuleList()
        self.st_layers.append(
            MST_RGCN_Layer(in_feats, h_feats, num_rels=11, dropout=dropout)
        )
        for _ in range(num_layers - 1):
            self.st_layers.append(
                MST_RGCN_Layer(h_feats, h_feats, num_rels=11, dropout=dropout)
            )

        # 预测用的MLP层
        final_input_dim = h_feats * num_layers * 2
        self.W1 = nn.Linear(final_input_dim, final_input_dim, bias = True)
        self.W2 = nn.Linear(final_input_dim, final_input_dim // 2, bias = True)
        self.W3 = nn.Linear(final_input_dim // 2, 1, bias = True)
        self.training = True

    # <--- MODIFIED: 添加 return_gate_values 标志
    def forward(self, g1, g2, g3, training, return_embed=False, return_gate_values=False):
        self.training = training
        
        features_list = [g1.ndata['feature'], g2.ndata['feature'], g3.ndata['feature']]
        graphs_list = [g1, g2, g3]
        num_nodes = g1.num_nodes()
        
        h_prev_t1_list = [
            torch.zeros(num_nodes, self.h_feats, device=self.device) for _ in range(self.num_layers)
        ]
        h_prev_t2_list = [
            torch.zeros(num_nodes, self.h_feats, device=self.device) for _ in range(self.num_layers)
        ]

        final_timestep_outputs = []
        
        # <--- MODIFIED: 用于存储门控值的列表
        gate_values_t = []

        # 按时间顺序处理每个图快照
        for t in range(len(graphs_list)):
            g_t = graphs_list[t]
            h_t_initial = features_list[t]
            
            h_in_layer = h_t_initial
            
            new_h_t_list = [] 
            skip_connection_features = []
            
            # <--- MODIFIED: 在每个时间步 t 开始时清空
            gate_values_t.clear()

            # 逐层通过MST-RGCN
            for i in range(self.num_layers):
                h_prev_t1 = h_prev_t1_list[i]
                h_prev_t2 = None if t == 0 else h_prev_t2_list[i]
                
                # <--- MODIFIED: 捕获 h_out_layer 和 gate
                h_out_layer, gate = self.st_layers[i](g_t, h_in_layer, h_prev_t1, h_prev_t2)
                
                # <--- MODIFIED: 如果需要，则存储门控值
                if return_gate_values and gate is not None:
                    gate_values_t.append(gate)
                
                new_h_t_list.append(h_out_layer)
                h_in_layer = h_out_layer
                skip_connection_features.append(h_out_layer)
            
            h_prev_t2_list = h_prev_t1_list
            h_prev_t1_list = new_h_t_list
            
            final_timestep_outputs.append(torch.cat(skip_connection_features, dim=1))
        
        output1, output2, output3 = final_timestep_outputs

        # 目标节点特征聚合
        enroll1, enroll2, enroll3 = (g.ndata['target'] == 1 for g in graphs_list)
        course1, course2, course3 = (g.ndata['target'] == 2 for g in graphs_list)
        
        em1 = torch.cat([output1[enroll1], output1[course1]], 1)
        em2 = torch.cat([output2[enroll2], output2[course2]], 1)
        em3 = torch.cat([output3[enroll3], output3[course3]], 1)
        
        final_repr = em3
        
        # 预测 MLP
        x = F.relu(self.W1(final_repr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.W2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W3(x)
        x = torch.sigmoid(x)
        
        
        if return_gate_values:
            # gate_values_t 此时包含最后时间步 t=2 的 L 个门控值
            # 我们返回第一层 (i=0) 的门控值，以匹配假设
            gate_to_return = gate_values_t[0] if len(gate_values_t) > 0 else None
            return x.reshape(-1), gate_to_return # 返回 (preds, gates)

        if return_embed:
            return x.reshape(-1), final_repr # 返回 (preds, embeds)
        
        return x.reshape(-1) # 仅返回 (preds)