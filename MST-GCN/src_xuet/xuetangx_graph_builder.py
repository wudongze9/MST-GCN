import os
import pandas as pd
import numpy as np
import dgl
import torch
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import pickle
from typing import Dict, List, Tuple

warnings.filterwarnings(action='ignore')
tqdm.pandas()

# =============================================================================
# 辅助函数：提取子图 (适配异构图)
# =============================================================================
def get_subgraph(graph: dgl.DGLGraph,
                 o_with_target: list,
                 e_with_student: list,
                 e_with_friend: list,
                 target_course: str,  # 原始 Course ID (String)
                 target_enroll: int,  # Enrollment DGL ID (Integer)
                 enroll_map: dict,
                 object_map: dict,
                 course_map: dict):
    """
    根据目标 Enrollment 及其关联节点，提取 k-hop 子图 (适配异构图)。
    """
    
    # 1. 收集所有相关的节点 (使用 DGL 内部 ID)
    nodes_to_keep = {
        'enrollment': set([target_enroll]),
        'object': set(),
        'course': set()
    }

    # 关联 Objects (使用 DGL ID)
    for oid in o_with_target:
        if oid in object_map:
            nodes_to_keep['object'].add(object_map[oid])
    
    # 关联 Courses (使用 DGL ID)
    if target_course in course_map:
        nodes_to_keep['course'].add(course_map[target_course])

    # 关联同学生/朋友的 Enrollments (使用 DGL ID)
    for eid_int in e_with_student + e_with_friend:
        nodes_to_keep['enrollment'].add(eid_int)

    # 过滤空集合
    nodes_dict = {
        ntype: torch.tensor(list(n_set), dtype=torch.long)
        for ntype, n_set in nodes_to_keep.items() if n_set
    }

    if not nodes_dict:
        return dgl.heterograph({})
        
    # 2. 提取子图
    subgraph = dgl.node_subgraph(graph, nodes_dict, store_ids=True)
    
    return subgraph

# =============================================================================
# 核心数据集类：XuetangXGraph (修复为异构图)
# =============================================================================
class XuetangXGraph(DGLDataset):
    """
    XuetangX 数据集的高性能构建类，已修复为构建 DGL 异构图并使用 Log 聚合特征。
    """
    def __init__(self, raw_dir=None, save_dir=None, subset_ratio=1.0, force_reload=False):
        self.subset_ratio = subset_ratio
        self.window_size = 35  # Dh = 35 days
        
        # 更改 name 以反映异构图和特征推断
        super(XuetangXGraph, self).__init__(name='xuetangx_hetero_inferred', 
                                             raw_dir=raw_dir,
                                             save_dir=save_dir if save_dir else '.',
                                             force_reload=force_reload)

    # ***************************************************************
    # 核心修复：process - 异构图构建和特征工程
    # ***************************************************************
    def process(self):
        print(f"Processing XuetangX Heterogeneous Graph (Subset Ratio: {self.subset_ratio})...")
        
        # 1. 加载数据 (使用修复后的 _load_raw_data)
        print("[1/5] Loading Raw Data...")
        logs, users, courses, truth_train = self._load_raw_data()
        
        # --- 采样逻辑 (保持不变) ---
        if self.subset_ratio < 1.0:
            np.random.seed(42)
            all_enrolls = truth_train['enroll_id'].unique()
            n_samples = int(len(all_enrolls) * self.subset_ratio)
            if n_samples > 0:
                print(f"Sampling {n_samples} enrollments...")
                sampled_eids = np.random.choice(all_enrolls, n_samples, replace=False)
                truth_train = truth_train[truth_train['enroll_id'].isin(sampled_eids)]
                logs = logs[logs['enroll_id'].isin(sampled_eids)]
                print(f"--- Subset created: {len(sampled_eids)} enrollments ---")

        # 仅保留窗口期内的日志
        logs = logs[logs['day'] < self.window_size].copy()

        # =================================================
        # 2. 向量化 ID 映射 (异构图的 ID 独立)
        # =================================================
        print("[2/5] Mapping IDs...")
        
        enroll_ids = truth_train['enroll_id'].unique()
        object_ids = logs['object'].unique()
        course_ids = courses['course_id'].unique()
        
        self.enroll_map = {eid: i for i, eid in enumerate(enroll_ids)}
        self.object_map = {oid: i for i, oid in enumerate(object_ids)}
        self.course_map = {cid: i for i, cid in enumerate(course_ids)}
        self.id_to_enroll = {v: k for k, v in self.enroll_map.items()}

        # =================================================
        # 3. 构建异构图
        # =================================================
        print("[3/5] Building Heterogeneous Graph Edges...")
        
        valid_logs = logs[logs['enroll_id'].isin(self.enroll_map) & logs['object'].isin(self.object_map)]
        
        # Enrollment <-> Object
        src_e_dgl = valid_logs['enroll_id'].map(self.enroll_map).values
        dst_o_dgl = valid_logs['object'].map(self.object_map).values
        eo_edges = pd.DataFrame({'s': src_e_dgl, 'd': dst_o_dgl}).drop_duplicates()
        src_e_dgl, dst_o_dgl = eo_edges['s'].values, eo_edges['d'].values
        
        # Object <-> Course
        obj_crs = logs[['object', 'course_id']].drop_duplicates()
        valid_oc = obj_crs[obj_crs['object'].isin(self.object_map) & obj_crs['course_id'].isin(self.course_map)]
        
        src_o_dgl = valid_oc['object'].map(self.object_map).values
        dst_c_dgl = valid_oc['course_id'].map(self.course_map).values

        # --- 使用 dgl.heterograph 构建图 ---
        data_dict = {
            ('enrollment', 'e_observes_o', 'object'): (torch.tensor(src_e_dgl), torch.tensor(dst_o_dgl)),
            ('object', 'o_observed_by_e', 'enrollment'): (torch.tensor(dst_o_dgl), torch.tensor(src_e_dgl)),
            ('object', 'o_belongs_to_c', 'course'): (torch.tensor(src_o_dgl), torch.tensor(dst_c_dgl)),
            ('course', 'c_contains_o', 'object'): (torch.tensor(dst_c_dgl), torch.tensor(src_o_dgl))
        }

        self.graph = dgl.heterograph(data_dict)
        print(f"Heterogeneous Graph built: {self.graph}")
        
        # =================================================
        # 4. 特征工程 (仅使用 Log 数据)
        # =================================================
        print("[4/5] Building Node Features from Log Data...")
        
        scaler = StandardScaler()
        
        # (A) Enrollment Features: 行为时间序列 + 注册实例聚合特征
        act_matrix = logs.groupby(['enroll_id', 'day']).size().unstack(fill_value=0)
        act_matrix = act_matrix.reindex(enroll_ids, fill_value=0)
        current_cols = act_matrix.shape[1]
        if current_cols < self.window_size:
            for c in range(current_cols, self.window_size):
                act_matrix[c] = 0
        act_values = act_matrix.iloc[:, :self.window_size].values
        
        enroll_action_dist = pd.crosstab(logs['enroll_id'], logs['action']).reset_index()
        enroll_action_dist = enroll_action_dist.set_index('enroll_id').reindex(enroll_ids).fillna(0)
        
        enroll_stats = logs.groupby('enroll_id').agg(
            total_actions=('action', 'size'),
            unique_objects=('object', 'nunique'),
        ).reindex(enroll_ids).fillna(0)
        
        enroll_agg_feat = pd.concat([enroll_stats, enroll_action_dist], axis=1).values
        enroll_agg_feat_scaled = scaler.fit_transform(enroll_agg_feat)
        enroll_feat = np.hstack([act_values, enroll_agg_feat_scaled])
        
        # (B) Course Features: 课程宏观行为聚合特征
        course_behavior = logs.groupby('course_id').agg(
            total_actions=('enroll_id', 'size'),
            num_enrollments=('enroll_id', 'nunique'),
            num_users=('username', 'nunique'),
        ).reset_index()
        
        course_behavior['density'] = course_behavior['total_actions'] / course_behavior['num_enrollments']
        
        courses_feat = courses.merge(course_behavior, on='course_id', how='left').fillna(0)
        courses_feat = courses_feat.set_index('course_id').reindex(course_ids)
        
        crs_feat_cols = [c for c in courses_feat.columns if c not in ['start', 'course_id']]
        crs_vec = courses_feat[crs_feat_cols].values
        
        crs_vec_scaled = scaler.fit_transform(crs_vec)
        self.num_course_feats = crs_vec_scaled.shape[1]
        
        # (C) Object Features: 对象行为分布特征
        obj_action_dist = pd.crosstab(logs['object'], logs['action']).reset_index()
        obj_action_dist = obj_action_dist.set_index('object').reindex(object_ids).fillna(0)
        obj_vec_scaled = scaler.fit_transform(obj_action_dist.values)
        
        # (D) 将特征分配给异构图的节点 (关键修复点)
        self.graph.nodes['enrollment'].data['feature'] = torch.tensor(enroll_feat, dtype=torch.float32)
        self.graph.nodes['object'].data['feature'] = torch.tensor(obj_vec_scaled, dtype=torch.float32)
        self.graph.nodes['course'].data['feature'] = torch.tensor(crs_vec_scaled, dtype=torch.float32)
        
        # 标签只针对 'enrollment' 节点
        truth_sorted = truth_train.set_index('enroll_id').reindex(enroll_ids)
        self.labels = torch.tensor(truth_sorted['truth'].values, dtype=torch.float32)

        # =================================================
        # 5. 构建查找索引 (不变)
        # =================================================
        print("[5/5] Building Adjacency Indices...")
        
        print("  > Grouping Enroll -> Objects...")
        self.enroll_obj_group = valid_logs.groupby('enroll_id')['object'].progress_apply(list).to_dict()
            
        print("  > Mapping Enroll -> Courses...")
        self.enroll_crs_map = truth_train.set_index('enroll_id')['course_id'].to_dict() 
        
        print("  > Grouping User -> Enrollments...")
        truth_train['eid_int'] = truth_train['enroll_id'].map(self.enroll_map)
        self.user_group = truth_train.groupby('username')['eid_int'].progress_apply(list).to_dict()
        self.enroll_user_map = truth_train.set_index('eid_int')['username'].to_dict()
        
        # Social Edges: friends.csv 缺失，friend_map 保持为空
        self.friend_map = self._load_friend_data(truth_train['username'].unique())


    # ***************************************************************
    # 核心修复：_load_raw_data - 移除虚拟数据创建
    # ***************************************************************
    def _load_raw_data(self):
        """
        加载真实数据，并执行关键的数据完整性修复 (仅推断必需的 course start time)。
        """
        path = self.raw_dir
        
        # 1. 加载 Logs (train + test)
        logs = pd.read_csv(os.path.join(path, 'train_log.csv'))
        if os.path.exists(os.path.join(path, 'test_log.csv')):
            print("Merging test logs...")
            test_logs = pd.read_csv(os.path.join(path, 'test_log.csv'))
            logs = pd.concat([logs, test_logs], ignore_index=True)
        
        # 2. 加载 Truth
        truth_train = pd.read_csv(os.path.join(path, 'train_truth.csv'))
        if os.path.exists(os.path.join(path, 'test_truth.csv')):
            truth_test = pd.read_csv(os.path.join(path, 'test_truth.csv'))
            truth_train = pd.concat([truth_train, truth_test], ignore_index=True)

        # Enrichment: 从 Logs 补充 username 和 course_id
        print("Enriching Truth data with metadata from Logs...")
        enroll_meta = logs[['enroll_id', 'username', 'course_id']].drop_duplicates(subset=['enroll_id'])
        truth_train = truth_train.merge(enroll_meta, on='enroll_id', how='left')
        truth_train = truth_train.dropna(subset=['username', 'course_id'])

        # 3. 处理缺失的 User Profile (不创建虚拟特征)
        unique_users = logs['username'].unique()
        users = pd.DataFrame({'user_id': unique_users, 'username': unique_users}) 
        
        # 4. 处理缺失的 Course Info (只推断 start time)
        unique_courses = logs['course_id'].unique()
        courses = pd.DataFrame({'course_id': unique_courses})
        
        print("Inferring course start times from logs (MANDATORY)...")
        course_starts = logs.groupby('course_id')['time'].min().reset_index()
        course_starts.columns = ['course_id', 'start']
        courses = courses.merge(course_starts, on='course_id', how='left')

        # 5. 计算 day (依赖于推断的 start time)
        print("Calculating 'day' column from timestamps (Vectorized)...")
        logs['time'] = pd.to_datetime(logs['time'])
        courses['start'] = pd.to_datetime(courses['start'])
        logs = logs.merge(courses[['course_id', 'start']], on='course_id', how='left')
        logs['day'] = (logs['time'] - logs['start']).dt.days
        logs.loc[logs['day'] < 0, 'day'] = 0

        return logs, users, courses, truth_train


    # ***************************************************************
    # 其他函数
    # ***************************************************************

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_dir, 'dgl_graph.bin'))

    def save(self):
        """保存图和辅助信息"""
        print("Saving graph to cache...")
        graph_path = os.path.join(self.save_dir, 'dgl_graph.bin')
        info_path = os.path.join(self.save_dir, 'info.pkl')
        
        # 异构图保存方式
        save_graphs(graph_path, [self.graph], {'labels': self.labels}) 
        
        with open(info_path, 'wb') as f:
            pickle.dump({
                'enroll_obj': self.enroll_obj_group,
                'enroll_crs': self.enroll_crs_map,
                'user_group': self.user_group,
                'enroll_user': self.enroll_user_map,
                'friend_map': self.friend_map,
                'id_to_enroll': self.id_to_enroll,
                'enroll_map': self.enroll_map,
                'object_map': self.object_map,
                'course_map': self.course_map,
                'num_course_feats': getattr(self, 'num_course_feats', 0)
            }, f)

    def load(self):
        """加载图和辅助信息"""
        print("Loading graph from cache...")
        graph_path = os.path.join(self.save_dir, 'dgl_graph.bin')
        info_path = os.path.join(self.save_dir, 'info.pkl')
        
        graphs, label_dict = load_graphs(graph_path)
        self.graph = graphs[0]
        self.labels = label_dict['labels']
        
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
            self.enroll_obj_group = info['enroll_obj']
            self.enroll_crs_map = info['enroll_crs']
            self.user_group = info['user_group']
            self.enroll_user_map = info['enroll_user']
            self.friend_map = info['friend_map']
            self.id_to_enroll = info['id_to_enroll']
            self.enroll_map = info['enroll_map']
            self.object_map = info['object_map']
            self.course_map = info['course_map']
            self.num_course_feats = info.get('num_course_feats', 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        target_eid_int = idx
        target_eid_raw = self.id_to_enroll[target_eid_int]
        
        objects_raw = self.enroll_obj_group.get(target_eid_raw, [])
        target_course_raw = self.enroll_crs_map.get(target_eid_raw, None)
        
        username = self.enroll_user_map.get(target_eid_int)
        e_with_student = self.user_group.get(username, [])
        
        e_with_friend = []
        if username and username in self.friend_map:
            friends = self.friend_map[username]
            for f in friends:
                if f in self.user_group:
                    e_with_friend.extend(self.user_group[f])
        
        subgraph = get_subgraph(
            self.graph,
            o_with_target=objects_raw,
            e_with_student=e_with_student,
            e_with_friend=e_with_friend,
            target_course=target_course_raw,
            target_enroll=target_eid_int,
            enroll_map=self.enroll_map,
            object_map=self.object_map,
            course_map=self.course_map
        )
        
        # 3. 设置 Target Mask (适配异构图)
        target_mask = torch.zeros(subgraph.num_nodes('enrollment'), dtype=torch.long)
        original_enroll_ids = subgraph.nodes['enrollment'].data[dgl.NID]
        target_mask[original_enroll_ids == target_eid_int] = 1
        subgraph.nodes['enrollment'].data['target'] = target_mask
        
        if 'course' in subgraph.ntypes:
            course_mask = torch.zeros(subgraph.num_nodes('course'), dtype=torch.long)
            target_course_dgl_id = self.course_map.get(target_course_raw)
            
            if target_course_dgl_id is not None:
                original_course_ids = subgraph.nodes['course'].data[dgl.NID]
                course_mask[original_course_ids == target_course_dgl_id] = 2
            
            subgraph.nodes['course'].data['target'] = course_mask
        
        if 'object' in subgraph.ntypes:
            subgraph.nodes['object'].data['target'] = torch.zeros(subgraph.num_nodes('object'), dtype=torch.long)

        # 返回 3 个子图以适配模型接口
        return subgraph, subgraph, subgraph, self.labels[idx].float()

    def _load_friend_data(self, users):
        friend_file = os.path.join(self.raw_dir, 'friends.csv')
        if os.path.exists(friend_file):
            print("Loading friends graph...")
            df = pd.read_csv(friend_file)
            return df.groupby('user1')['user2'].apply(list).to_dict()
        print("WARNING: 'friends.csv' not found. Social edges are ignored.")
        return {}