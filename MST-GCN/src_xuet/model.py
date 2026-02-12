import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn import HeteroGraphConv, GraphConv, GATConv
import math


class MST_GNN_Layer(nn.Module):
    """
    Multi-Scale Spatio-Temporal Graph Convolutional Layer.
    """
    # 修复: 仅接受核心参数 in_feats, out_feats, dropout
    def __init__(self, in_feats, out_feats, dropout=0.2): 
        super(MST_GNN_Layer, self).__init__()
        
        self.rel_names = [
            ('course', 'c_contains_o', 'object'),
            ('object', 'o_belongs_to_c', 'course'),
            ('object', 'o_observed_by_e', 'enrollment'),
            ('enrollment', 'e_observes_o', 'object'),
        ]
        
        # 空间聚合层: GraphConv 中移除了 allow_zero_in
        self.hgraph_conv = HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats, norm='both') 
            for rel in self.rel_names
        }, aggregate='sum')
        
        self.gru = nn.GRUCell(input_size=out_feats, hidden_size=out_feats)
        self.gate_linear = nn.Linear(out_feats * 3, out_feats)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_feats = out_feats

    # ** 核心修复位于此函数 **
    def forward(self, g, h_in_dict, prev_h_t1_dict, prev_h_t2_dict):
        spatial_msg_dict = self.hgraph_conv(g, h_in_dict)
        h_out_dict = {}
        gate_dict = {}
        
        for ntype, h_spatial in spatial_msg_dict.items():
            
            # 1. 检查当前节点类型是否是需要GRU更新的节点 (即具有历史状态)
            if ntype not in prev_h_t1_dict:
                h_out_dict[ntype] = self.activation(h_spatial)
                gate_dict[ntype] = None
                continue

            prev_h_t1 = prev_h_t1_dict[ntype]
            
            # 2. 修复 NoneType 错误：先检查 prev_h_t2_dict 是否为 None
            if prev_h_t2_dict is None:
                prev_h_t2 = None
            else:
                prev_h_t2 = prev_h_t2_dict.get(ntype) # 安全地从字典中获取
            
            # 3. 多尺度时间融合
            if prev_h_t2 is None:
                fused_h = prev_h_t1
                gate = None
            else:
                gate_input = torch.cat([h_spatial, prev_h_t1, prev_h_t2], dim=1)
                gate = torch.sigmoid(self.gate_linear(gate_input))
                fused_h = gate * prev_h_t1 + (1 - gate) * prev_h_t2

            # 4. 时间更新 (GRU)
            h_t = self.gru(h_spatial, fused_h)
            h_t = self.activation(h_t)
            h_out_dict[ntype] = self.dropout(h_t)
            gate_dict[ntype] = gate

        return h_out_dict, gate_dict


class Multi_MST_GCN(nn.Module):
    def __init__(self, num_layers, 
                 enroll_in_feats, 
                 object_in_feats, 
                 course_in_feats, 
                 h_feats, 
                 bi, 
                 dropout, 
                 batch_size, 
                 device):
        super(Multi_MST_GCN, self).__init__()
        self.num_layers = num_layers
        self.h_feats = h_feats
        self.dropout = dropout
        self.device = device
        self.ntypes = ['enrollment', 'object', 'course']

        self.proj_e = nn.Linear(enroll_in_feats, h_feats)
        self.proj_o = nn.Linear(object_in_feats, h_feats)
        self.proj_c = nn.Linear(course_in_feats, h_feats)

        self.st_layers = nn.ModuleList()
        self.st_layers.append(MST_GNN_Layer(h_feats, h_feats, dropout=dropout)) 
        for _ in range(num_layers - 1):
            self.st_layers.append(MST_GNN_Layer(h_feats, h_feats, dropout=dropout))

        # 预测用的MLP层
        # Enrollment: h_feats * num_layers, Course: h_feats * num_layers
        final_input_dim = h_feats * num_layers * 2 
        self.W1 = nn.Linear(final_input_dim, final_input_dim, bias = True)
        self.W2 = nn.Linear(final_input_dim, final_input_dim // 2, bias = True)
        self.W3 = nn.Linear(final_input_dim // 2, 1, bias = True)
        self.training = True

    def forward(self, g1, g2, g3, training, return_embed=False, return_gate_values=False):
        self.training = training
        graphs_list = [g1, g2, g3]
        
        def project_features(g):
            h_e = F.relu(self.proj_e(g.nodes['enrollment'].data['feature']))
            h_o = F.relu(self.proj_o(g.nodes['object'].data['feature']))
            h_c = F.relu(self.proj_c(g.nodes['course'].data['feature']))
            return {'enrollment': h_e, 'object': h_o, 'course': h_c}
            
        initial_h_list = [project_features(g) for g in graphs_list]
        
        # 2. 初始化历史状态字典
        g1_e_nodes = g1.num_nodes('enrollment')
        g1_o_nodes = g1.num_nodes('object')
        g1_c_nodes = g1.num_nodes('course')

        zero_h_dict = {
            'enrollment': torch.zeros(g1_e_nodes, self.h_feats, device=self.device), 
            'object': torch.zeros(g1_o_nodes, self.h_feats, device=self.device), 
            'course': torch.zeros(g1_c_nodes, self.h_feats, device=self.device)
        }
        
        h_prev_t1_list = [zero_h_dict.copy() for _ in range(self.num_layers)]
        h_prev_t2_list = [zero_h_dict.copy() for _ in range(self.num_layers)]

        # 修改了输出存储结构：只存储最后时间步 t=2 的拼接特征
        final_enroll_output = None 
        final_course_output = None 
        
        gate_values_t = []

        # 3. 按时间顺序处理每个图快照
        for t in range(len(graphs_list)):
            g_t = graphs_list[t]
            h_in_dict = initial_h_list[t]
            
            new_h_t_list = []
            skip_e_features = []  # Enrollment skip features
            skip_c_features = []  # Course skip features

            gate_values_t.clear()

            for i in range(self.num_layers):
                h_prev_t1_dict = h_prev_t1_list[i]
                h_prev_t2_dict = None if t == 0 else h_prev_t2_list[i]

                h_out_dict, gate_dict = self.st_layers[i](g_t, h_in_dict, h_prev_t1_dict, h_prev_t2_dict)
                
                h_out_enroll = h_out_dict['enrollment']
                h_out_course = h_out_dict['course'] # 获取 Course 节点特征
                
                gate_enroll = gate_dict['enrollment']
                
                if return_gate_values and gate_enroll is not None:
                    gate_values_t.append(gate_enroll)
                
                h_in_dict = h_out_dict
                new_h_t_list.append(h_out_dict)
                
                # 存储跳跃连接特征
                skip_e_features.append(h_out_enroll)
                skip_c_features.append(h_out_course) # 存储 Course 跳跃特征
            
            h_prev_t2_list = h_prev_t1_list
            h_prev_t1_list = new_h_t_list
            
            # 仅在最后一个时间步 (t=2) 存储拼接特征
            if t == 2:
                final_enroll_output = torch.cat(skip_e_features, dim=1)
                final_course_output = torch.cat(skip_c_features, dim=1)

        # 4. 目标节点特征聚合 (使用最后一个时间步 g3)
        g3_enroll_mask = g3.nodes['enrollment'].data['target'] == 1
        g3_course_mask = g3.nodes['course'].data['target'] == 2

        # 从 Enrollment 特征中提取目标
        em_enroll = final_enroll_output[g3_enroll_mask]
        
        # 从 Course 特征中提取目标
        em_course = final_course_output[g3_course_mask]

        # 确保 em_enroll 和 em_course 维度一致后再拼接
        if em_enroll.size(0) == em_course.size(0):
            final_repr = torch.cat([em_enroll, em_course], dim=1)
        else:
             min_size = min(em_enroll.size(0), em_course.size(0))
             final_repr = torch.cat([em_enroll[:min_size], em_course[:min_size]], dim=1)
        
        # 预测 MLP
        x = F.relu(self.W1(final_repr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.W2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W3(x)
        preds = torch.sigmoid(x)
        
        # 5. 返回逻辑
        if return_gate_values:
            gate_to_return = gate_values_t[0] if len(gate_values_t) > 0 else None
            return preds.reshape(-1), gate_to_return 

        if return_embed:
            return preds.reshape(-1), final_repr 
        
        return preds.reshape(-1)


