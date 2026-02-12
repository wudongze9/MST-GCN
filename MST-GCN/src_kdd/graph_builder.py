import pandas as pd
import numpy as np
import dgl
from dgl.data import DGLDataset
import torch
from tqdm import tqdm
import warnings
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import random
from itertools import permutations 
from utils import kddcup_load

warnings.filterwarnings(action='ignore')


def get_subgraph(graph: dgl.graph,
                 o_with_target: torch.tensor,
                 e_with_student: torch.tensor,
                 target_course: torch.tensor,
                 target_enroll: torch.tensor,
                 ):
    nodes = torch.unique(torch.cat([o_with_target, e_with_student, target_course, target_enroll], dim=0,))
    nodes = nodes.type(torch.int64)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True)
    return subgraph

class kddcup_graph(DGLDataset):
    def __init__(self, subset_ratio=1.0, window_size=10):
        self.subset_ratio = subset_ratio
        self.window_size = window_size
        self.num_snapshots = 3
        self.total_days = self.window_size * self.num_snapshots
        super().__init__(name='kddcup')
        
    def process(self):
        # 1. 数据加载
        print('Loading data...')
        course_date, course_info, enrollment_train, enrollment_test, log_train, log_test, truth_train, truth_test, truth_all, log_all, enrollment_all = kddcup_load()
        
        # --- 数据采样逻辑 ---
        if self.subset_ratio < 1.0:
            np.random.seed(0)
            random.seed(0)
            all_enrollment_ids = truth_all['enrollment_id'].unique()
            num_samples = int(len(all_enrollment_ids) * self.subset_ratio)
            if num_samples == 0: num_samples = 1
            sampled_enrollment_ids = np.random.choice(all_enrollment_ids, size=num_samples, replace=False)
            
            truth_all = truth_all[truth_all['enrollment_id'].isin(sampled_enrollment_ids)]
            log_all = log_all[log_all['enrollment_id'].isin(sampled_enrollment_ids)]
            enrollment_all = enrollment_all[enrollment_all['enrollment_id'].isin(sampled_enrollment_ids)]
            truth_train = truth_train[truth_train['enrollment_id'].isin(sampled_enrollment_ids)]
            truth_test = truth_test[truth_test['enrollment_id'].isin(sampled_enrollment_ids)]
            
            sampled_course_ids = enrollment_all['course_id'].unique()
            course_date = course_date[course_date['course_id'].isin(sampled_course_ids)]
            course_info = course_info[course_info['course_id'].isin(sampled_course_ids)]
            print(f"--- Data constrained to {len(sampled_enrollment_ids)} enrollment records ({self.subset_ratio*100:.2f}%) ---")


        log_all['date_dt'] = pd.to_datetime(log_all['time'].str.split('T').str[0], format='%Y-%m-%d')
        # 合并 enrollment 获取 start time
        log_all = pd.merge(log_all, enrollment_all[['enrollment_id', 'course_id']], on='enrollment_id', how='left')
        log_all = pd.merge(log_all, course_date, on='course_id', how='left')
        
        log_all['from_dt'] = pd.to_datetime(log_all['from'], format='%Y-%m-%d')
        log_all['days'] = (log_all['date_dt'] - log_all['from_dt']).dt.days

        truth_all = pd.merge(truth_all, enrollment_all, how='left', on='enrollment_id')

        # Event Mapping 
        event_map = {'problem': 0, 'video': 1, 'discussion': 2, 'wiki': 3, 'page_close': 3, 'navigate': 3, 'access': 3}
        log_all['event'] = log_all['event'].map(event_map).fillna(3).astype(int) # fillna以防万一

        # Groupby counting (Keep this logic)
        all_clicksum = log_all.groupby(['enrollment_id', 'event', 'days', 'object']).size().reset_index(name='click_sum')

        # Split days
        # 这里不需要显式 copy 整个 dataframe，可以用 mask
        mask_one = log_all['days'] < self.window_size
        mask_two = (log_all['days'] >= self.window_size) & (log_all['days'] < self.window_size * 2)
        mask_third = (log_all['days'] >= self.window_size * 2) & (log_all['days'] < self.total_days)
        
        activity = [0, 1, 2, 3]
        log_activity = log_all[log_all['event'].isin(activity)] # 基础 activity log

        enroll_list = sorted(list(set(truth_all['enrollment_id'])))
        object_list = sorted(list(set(log_activity['object'])))
        course_list = sorted(list(set(course_date['course_id'])))

        # 建立映射字典
        enroll_node_id = {eid: i for i, eid in enumerate(enroll_list)}
        object_node_id = {oid: i + len(enroll_list) for i, oid in enumerate(object_list)}
        course_node_id = {cid: i + len(enroll_list) + len(object_list) for i, cid in enumerate(course_list)}

        enroll_num_node = len(enroll_node_id)
        object_num_node = len(object_node_id)
        course_num_node = len(course_node_id)
        num_node = enroll_num_node + object_num_node + course_num_node

        # 预先将 DataFrame 中的 ID 转换为 Node ID，避免后续循环 map
        truth_all['eid_mapped'] = truth_all['enrollment_id'].map(enroll_node_id)
        truth_all['cid_mapped'] = truth_all['course_id'].map(course_node_id)
        

        truth_lookup = truth_all.set_index('eid_mapped')['truth'].reindex(range(len(enroll_list))).fillna(0).values


        print('Feature extraction (Vectorized)...')
        
        all_clicksum['eid_mapped'] = all_clicksum['enrollment_id'].map(enroll_node_id)

        clicksum_valid = all_clicksum[all_clicksum['days'] < self.total_days]
        
        # 创建稀疏矩阵或密集矩阵 (Pivot)
        feature_matrix_df = clicksum_valid.pivot_table(
            index='eid_mapped', columns='days', values='click_sum', aggfunc='sum'
        ).reindex(index=range(len(enroll_list)), columns=range(self.total_days)).fillna(0)
        
        enroll_feature_matrix = feature_matrix_df.values # Shape: (N_enroll, Total_Days)
        
        # Label 生成 (Vectorized)
        labels0 = (enroll_feature_matrix[:, 0:self.window_size].sum(axis=1) == 0).astype(int)
        labels1 = (enroll_feature_matrix[:, self.window_size:self.window_size*2].sum(axis=1) == 0).astype(int)
        labels2 = (enroll_feature_matrix[:, self.window_size*2:self.total_days].sum(axis=1) == 0).astype(int)
        labels3 = truth_lookup.astype(int)

        labels_df = pd.DataFrame({
            'enroll_id': enroll_list,
            'labels0': labels0, 'labels1': labels1, 'labels2': labels2, 'labels3': labels3
        })

        # Object Feature
        object_event_counts = log_activity.groupby(['object', 'event']).size().reset_index()
        object_event_counts['oid_mapped'] = object_event_counts['object'].map(object_node_id)
        
        object_feature_matrix = np.zeros((len(object_list), 4)) # 4 activities

        obj_indices = (object_event_counts['oid_mapped'] - enroll_num_node).values.astype(int)
        evt_indices = object_event_counts['event'].values.astype(int)
        object_feature_matrix[obj_indices, evt_indices] = 1
        
        # Course Feature
        course_cate_list = ['video', 'problem', 'discussion', 'chapter', 'html', 'about']
        course_cate_to_idx = {c: i for i, c in enumerate(course_cate_list)}
        
        course_info_valid = course_info[course_info['category'].isin(course_cate_list)].copy()
        course_info_valid['cid_mapped'] = course_info_valid['course_id'].map(course_node_id)
        course_info_valid['cate_idx'] = course_info_valid['category'].map(course_cate_to_idx)
        
        # 统计每个课程各类别的数量
        course_cate_counts = course_info_valid.groupby(['cid_mapped', 'cate_idx']).size().reset_index(name='count')
        
        course_feature_matrix = np.zeros((len(course_list), len(course_cate_list) + 1))
        
        # 填充类别计数
        c_rows = (course_cate_counts['cid_mapped'] - enroll_num_node - object_num_node).values.astype(int)
        c_cols = course_cate_counts['cate_idx'].values.astype(int)
        course_feature_matrix[c_rows, c_cols] = course_cate_counts['count'].values
        
        # 填充 enrollment 数量 (最后一列)
        enroll_counts = truth_all['cid_mapped'].value_counts()
        for cid_map, count in enroll_counts.items():
            idx = cid_map - enroll_num_node - object_num_node
            if 0 <= idx < len(course_list):
                course_feature_matrix[idx, -1] = count

        # Standard Scaler
        stan_scaler = StandardScaler()
        

        
        col_name = [f'clicksum_day{i}' for i in range(self.total_days)]
        temp_enroll_df = pd.DataFrame(enroll_feature_matrix, columns=col_name)
        temp_enroll_df['enrollment_id'] = enroll_list
        temp_enroll_df = pd.merge(temp_enroll_df, enrollment_all[['enrollment_id', 'course_id']], on='enrollment_id', how='left')
        
        # Groupby transform for standardization
        features_only = temp_enroll_df[col_name]

        node_feature_df_list = []
        for course_id, group in temp_enroll_df.groupby('course_id'):
             if not group.empty:
                 feat = group[col_name]
                 scaled = stan_scaler.fit_transform(feat)
                 scaled_df = pd.DataFrame(scaled, index=group['enrollment_id'], columns=col_name)
                 node_feature_df_list.append(scaled_df)
        
        node_feature_df = pd.concat(node_feature_df_list)
        # 重新排序以匹配 enroll_node_id 顺序 (map enroll_node_id to index)
        node_feature_df = node_feature_df.reindex(enroll_list).fillna(0) # 确保顺序一致
        
        # Course feature standardization
        course_feature_scaled = stan_scaler.fit_transform(course_feature_matrix)


        print('Graph building (Vectorized)...')
        
        # 辅助结构构建
        e_in_course_dic = defaultdict(list)
        o_in_course_dic = defaultdict(list)
        
        # 快速构建字典
        # Object -> Course
        obj_course_df = pd.merge(log_activity[['object']].drop_duplicates(), 
                                 pd.merge(log_activity[['object', 'course_id']].drop_duplicates(), course_date[['course_id']], on='course_id'), 
                                 on='object', how='inner')

        object_course_map = log_all[['object', 'course_id']].drop_duplicates()
        object_course_map = object_course_map[object_course_map['object'].isin(object_list)]
        
        for _, row in object_course_map.iterrows(): 
            if row['course_id'] in course_node_id and row['object'] in object_node_id:
                o_in_course_dic[course_node_id[row['course_id']]].append(object_node_id[row['object']])
        
        # Enrollment -> Course
        for cid_mapped, grp in truth_all.groupby('cid_mapped'):
            e_in_course_dic[cid_mapped] = grp['eid_mapped'].tolist()

        oc_edges = pd.merge(object_event_counts, object_course_map, on='object', how='left')
        oc_edges['src'] = oc_edges['oid_mapped']
        oc_edges['dst'] = oc_edges['course_id'].map(course_node_id)
        oc_edges = oc_edges.dropna(subset=['dst']) # 过滤无效
        oc_edges['dst'] = oc_edges['dst'].astype(int)
        
        # Event type mapping is identity for 0,1,2,3
        oc_src = oc_edges['src'].tolist()
        oc_dst = oc_edges['dst'].tolist()
        oc_etype = oc_edges['event'].tolist()

        # --- 2. Enrollment-Object Edges (Per snapshot) ---
        def build_interaction_edges(log_subset, offset_etype):
            # log_subset has columns: enrollment_id, object, event
            # 聚合 unique interactions
            interactions = log_subset.groupby(['enrollment_id', 'object', 'event']).size().reset_index()
            interactions['src'] = interactions['enrollment_id'].map(enroll_node_id)
            interactions['dst'] = interactions['object'].map(object_node_id)
            # Filter NaNs if any
            interactions = interactions.dropna(subset=['src', 'dst'])
            
            src = interactions['src'].astype(int).tolist()
            dst = interactions['dst'].astype(int).tolist()
            # etype: problem(0)->4, video(1)->5, discuss(2)->6, wiki(3)->7
            etype = (interactions['event'] + 4).tolist()
            
            # Bidirectional
            return src + dst, dst + src, etype + etype, interactions

        log_one = log_activity[mask_one]
        log_two = log_activity[mask_two]
        log_three = log_activity[mask_third]
        
        one_src, one_dst, one_etype, one_inter_df = build_interaction_edges(log_one, 0)
        two_src, two_dst, two_etype, two_inter_df = build_interaction_edges(log_two, 0)
        third_src, third_dst, third_etype, third_inter_df = build_interaction_edges(log_three, 0)
        
        
        def build_lookup_dicts(inter_df):
            e_in_obj = defaultdict(list)
            o_in_enroll = defaultdict(list)
            # 使用 zip 迭代比 iterrows 快
            for e, o in zip(inter_df['src'], inter_df['dst']):
                e_in_obj[o].append(e)
                o_in_enroll[e].append(o)
            return e_in_obj, o_in_enroll

        self.one_e_in_object_dic, self.one_o_in_enroll_dic = build_lookup_dicts(one_inter_df)
        self.two_e_in_object_dic, self.two_o_in_enroll_dic = build_lookup_dicts(two_inter_df)
        self.third_e_in_object_dic, self.third_o_in_enroll_dic = build_lookup_dicts(third_inter_df)

        # 合并 Object-Course 边
        one_src.extend(oc_src); one_dst.extend(oc_dst); one_etype.extend(oc_etype)
        two_src.extend(oc_src); two_dst.extend(oc_dst); two_etype.extend(oc_etype)
        third_src.extend(oc_src); third_dst.extend(oc_dst); third_etype.extend(oc_etype)

        student_groups = truth_all.groupby('username')['eid_mapped'].apply(list).to_dict()
        
        # 准备 label 查找数组 (Index -> Label)
        # labels1 array created earlier corresponds to enroll_list order (0 to N)
        lab1_arr = labels1
        lab2_arr = labels2
        lab3_arr = labels3 
        
        s_edges_src = []
        s_edges_dst = []
        s_etype1 = []
        s_etype2 = []
        s_etype3 = []
        
        student_eid_dic_1 = defaultdict(list)
        student_eid_dic_2 = defaultdict(list)
        student_eid_dic_3 = defaultdict(list)
        
        # 当前边数量偏移量 (用于记录 student edge 的 ID)
        offset_1 = len(one_src)
        offset_2 = len(two_src)
        offset_3 = len(third_src)

        for student, eids in tqdm(student_groups.items(), desc="Building Student Edges"):
            if len(eids) < 2: continue
            
            # 生成全排列边 (src, dst)
            perms = list(permutations(eids, 2))
            srcs = [p[0] for p in perms]
            dsts = [p[1] for p in perms]
            
            count = len(srcs)
            s_edges_src.extend(srcs)
            s_edges_dst.extend(dsts)
            
            # 批量计算 etype
            # Src label = 1 (dropout) -> etype 8, else 9
            # Label1
            src_l1 = lab1_arr[srcs]
            et1 = np.where(src_l1 == 1, 8, 9)
            s_etype1.extend(et1)
            
            # Label2
            src_l2 = lab2_arr[srcs]
            et2 = np.where(src_l2 == 1, 8, 9)
            s_etype2.extend(et2)
            
            # Label3 (Truth)
            src_l3 = lab3_arr[srcs]
            et3 = np.where(src_l3 == 1, 8, 9)
            s_etype3.extend(et3)
            
            # 记录 Edge ID (用于 Masking)
            # 这部分比较棘手，因为我们需要记录每条边对应的ID。
            # 当前 range: [offset, offset + count)
            edge_ids_1 = list(range(offset_1, offset_1 + count))
            edge_ids_2 = list(range(offset_2, offset_2 + count))
            edge_ids_3 = list(range(offset_3, offset_3 + count))
            
            # 将这些 ID 分配给对应的 Source Student (Enrollment ID)
            # perms 顺序是 (src0, dst1), (src0, dst2)...
            for i, src_id in enumerate(srcs):
                student_eid_dic_1[src_id].append(edge_ids_1[i])
                student_eid_dic_2[src_id].append(edge_ids_2[i])
                student_eid_dic_3[src_id].append(edge_ids_3[i])
            
            offset_1 += count
            offset_2 += count
            offset_3 += count

        # 添加 Student Edges 到主列表
        one_src.extend(s_edges_src); one_dst.extend(s_edges_dst); one_etype.extend(s_etype1)
        two_src.extend(s_edges_src); two_dst.extend(s_edges_dst); two_etype.extend(s_etype2)
        third_src.extend(s_edges_src); third_dst.extend(s_edges_dst); third_etype.extend(s_etype3)
        
        self.one_student_eid_dic = student_eid_dic_1
        self.two_student_eid_dic = student_eid_dic_2
        self.third_student_eid_dic = student_eid_dic_3
        self.e_with_student_dic = student_groups

        # --- 4. 构建 DGL 图 ---
        one_graph = dgl.graph((one_src, one_dst), num_nodes=num_node)
        two_graph = dgl.graph((two_src, two_dst), num_nodes=num_node)
        third_graph = dgl.graph((third_src, third_dst), num_nodes=num_node)

        # --- 5. 组装节点特征 (Vectorized) ---
        print('Assembling node features...')
        
        # 构造全零/Padding 向量
        # Enroll Features: [1, 0, 0, clicksum(window), 0...0(course)]
        # Object Features: [0, 1, 0, 0...0(clicksum), 0...0(course)]
        # Course Features: [0, 0, 1, 0...0(clicksum), counts...]
        
        zero_enroll_window = np.zeros((1, self.window_size))
        zero_course_feat = np.zeros((1, len(course_cate_list) + 1))
        
        # 预先分配整个特征矩阵 (Num_Node, Feature_Dim)
        # Feature Dim = 3 (Type) + Window_Size + Course_Feat_Len
        feat_dim = 3 + self.window_size + (len(course_cate_list) + 1)
        
        def build_feature_tensor(graph, time_idx):
            # time_idx: 0, 1, 2 for one, two, third slices
            
            # --- Enrollment Nodes ---
            # Type: [1, 0, 0]
            e_type = np.tile([1, 0, 0], (enroll_num_node, 1))
            # Clicks: slice from node_feature_df
            # node_feature_df 已经按 index 0..N 排序且标准化
            start = time_idx * self.window_size
            end = (time_idx + 1) * self.window_size
            # 这里的 node_feature_df 是 DataFrame，转 numpy
            e_clicks = node_feature_df.iloc[:, start:end].values 
            # Course Part: 0
            e_course = np.zeros((enroll_num_node, zero_course_feat.shape[1]))
            
            e_feat = np.hstack([e_type, e_clicks, e_course])
            
            # --- Object Nodes ---
            # Type: [0, 1, 0]
            o_type = np.tile([0, 1, 0], (object_num_node, 1))
            o_clicks = np.zeros((object_num_node, self.window_size))
            o_course = np.zeros((object_num_node, zero_course_feat.shape[1]))
            o_feat = np.hstack([o_type, o_clicks, o_course])
            
            # --- Course Nodes ---
            # Type: [0, 0, 1]
            c_type = np.tile([0, 0, 1], (course_num_node, 1))
            c_clicks = np.zeros((course_num_node, self.window_size))
            # Course Feat (Scaled)
            c_course = course_feature_scaled
            c_feat = np.hstack([c_type, c_clicks, c_course])
            
            # Stack all: Enroll -> Object -> Course
            full_feat = np.vstack([e_feat, o_feat, c_feat])
            return torch.FloatTensor(full_feat)

        one_graph.ndata['feature'] = build_feature_tensor(one_graph, 0)
        two_graph.ndata['feature'] = build_feature_tensor(two_graph, 1)
        third_graph.ndata['feature'] = build_feature_tensor(third_graph, 2)
        
        # Non-feature (One-hot Type)
        non_feat_e = np.tile([1, 0, 0], (enroll_num_node, 1))
        non_feat_o = np.tile([0, 1, 0], (object_num_node, 1))
        non_feat_c = np.tile([0, 0, 1], (course_num_node, 1))
        full_non_feat = torch.FloatTensor(np.vstack([non_feat_e, non_feat_o, non_feat_c]))
        
        one_graph.ndata['non_feature'] = full_non_feat
        two_graph.ndata['non_feature'] = full_non_feat
        third_graph.ndata['non_feature'] = full_non_feat
        
        one_graph.edata['etype'] = torch.tensor(one_etype)
        two_graph.edata['etype'] = torch.tensor(two_etype)
        third_graph.edata['etype'] = torch.tensor(third_etype)

        # 保存结果
        self.one_graph = one_graph
        self.two_graph = two_graph
        self.third_graph = third_graph
        self.truth_all = truth_all
        self.labels_df = labels_df
        self.enroll_node_id = enroll_node_id
        self.object_node_id = object_node_id
        self.course_node_id = course_node_id
        self.e_in_cousre_dic = e_in_course_dic
        self.o_in_cousre_dic = o_in_course_dic
        # e_with_student_dic 和 student_eid_dic 已在上面赋值
        
        print("Graph processing complete.")

    def __len__(self):
        return len(self.labels_df) + 1
        
    def __getitem__(self, i):
        # 逻辑完全保持不变
        if i == 0:
            return self.one_graph
            
        try:
            target_enroll_id = self.labels_df.iloc[i - 1]['enroll_id']
        except IndexError:
            raise IndexError(f"Index {i} out of bounds.")
        
        target_enroll = self.enroll_node_id[target_enroll_id]
        # 注意：truth_all 已经有了 mapped columns，但为了兼容旧逻辑我们用 node_id dict
        enroll_info = self.truth_all[self.truth_all['enrollment_id'] == target_enroll_id]
        
        target_course = self.course_node_id[enroll_info['course_id'].values[0]]
        student_name = enroll_info['username'].values[0]
        
        o_with_target = self.o_in_cousre_dic[target_course]
        enroll_list = self.e_with_student_dic.get(student_name, [])
        
        # 提取子图
        args = {
            'o_with_target': torch.tensor(o_with_target),
            'e_with_student': torch.tensor(enroll_list, dtype=torch.int64),
            'target_course': torch.tensor([target_course], dtype=torch.int64),
            'target_enroll': torch.tensor([target_enroll], dtype=torch.int64),
        }
        
        one_subgraph = get_subgraph(self.one_graph, **args)
        two_subgraph = get_subgraph(self.two_graph, **args)
        third_subgraph = get_subgraph(self.third_graph, **args)
        
        # Target Masking
        def apply_target_mask(subgraph):
            target_mask = np.zeros(subgraph.num_nodes())
            target_enroll_id_subgraph = (subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
            target_course_id_subgraph = (subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
            if len(target_enroll_id_subgraph) > 0: target_mask[target_enroll_id_subgraph] = 1
            if len(target_course_id_subgraph) > 0: target_mask[target_course_id_subgraph] = 2
            subgraph.ndata['target'] = torch.tensor(target_mask)
            return subgraph

        one_subgraph = apply_target_mask(one_subgraph)
        two_subgraph = apply_target_mask(two_subgraph)
        third_subgraph = apply_target_mask(third_subgraph)
        
        # Edge Masking (Student Edge Leakage Prevention)
        def mask_student_edges(subgraph, edge_ids_list):
            if not edge_ids_list: return
            
            target_enroll_id_subgraph = (subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
            if len(target_enroll_id_subgraph) == 0: return # Target node might be filtered out? Unlikely but safe.

            # 找到 subgraph 中对应全局 edge_id 的边索引
            # 这是一个比较慢的操作：subgraph.edata[dgl.EID] 查找
            # 优化：只查找属于 target_enroll 的边
            sg_eids = subgraph.edata[dgl.EID]
            # Convert list to tensor for comparison
            drop_candidates = torch.tensor(edge_ids_list, dtype=sg_eids.dtype)
            
            # Find indices in subgraph where EID is in drop_candidates
            # isin available in newer pytorch, or use numpy
            mask = torch.isin(sg_eids, drop_candidates)
            drop_etype_idx = mask.nonzero(as_tuple=True)[0]
            
            if len(drop_etype_idx) == 0: return

            # 获取这些边的另一端节点 (dst)
            u, v = subgraph.edges()
            
            e_list = v[drop_etype_idx] # dst nodes of the dropped edges
            
            # 寻找反向边: src 是 e_list 中的点，dst 是 target_enroll
            # 这是一个 naive loop，但由于子图很小，所以是可以接受的
            new_etype_indices = []
            valid_drop_indices = []
            
            target_node_idx = target_enroll_id_subgraph.item()
            
            for i, node_idx in enumerate(e_list):
                # 查找边: src=node_idx, dst=target_node_idx
                # (u == node_idx) & (v == target_node_idx)
                edge_idx = ((u == node_idx) & (v == target_node_idx)).nonzero(as_tuple=True)[0]
                if len(edge_idx) > 0:
                    new_etype_indices.append(edge_idx[0].item())
                    valid_drop_indices.append(drop_etype_idx[i].item())
            
            if len(valid_drop_indices) > 0:
                 subgraph.edata['etype'][valid_drop_indices] = subgraph.edata['etype'][new_etype_indices]

        mask_student_edges(one_subgraph, self.one_student_eid_dic[target_enroll])
        mask_student_edges(two_subgraph, self.two_student_eid_dic[target_enroll])
        mask_student_edges(third_subgraph, self.third_student_eid_dic[target_enroll])

        labels_row = self.labels_df.iloc[i-1]
        return one_subgraph, two_subgraph, third_subgraph, torch.Tensor([labels_row['labels0'], labels_row['labels1'], labels_row['labels2'], labels_row['labels3']])