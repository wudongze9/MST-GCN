# Helper for dynamic import
from importlib import import_module
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import dgl
from torch.optim import Adam
from model import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
import inspect


def custom_collate(samples):
    # 过滤掉非元组数据 (防止 Index 0 的 DGLGraph 混入)
    valid_samples = [s for s in samples if isinstance(s, tuple) and len(s) == 4]
    
    if not valid_samples:
        return None, None, None, None

    g1_list, g2_list, g3_list, labels = zip(*valid_samples)
    batched_g1 = dgl.batch(g1_list)
    batched_g2 = dgl.batch(g2_list)
    batched_g3 = dgl.batch(g3_list)
    batched_labels = torch.stack(labels)
    return batched_g1, batched_g2, batched_g3, batched_labels


def eval(model, test_dataloader, device):
    """
    评估函数：计算最终测试指标并返回详细报告。
    """
    model.eval()
    val_labels = []
    val_preds = []
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            g1, g2, g3, labels = batch
            if g1 is None: continue # 处理无效 batch
            
            g1, g2, g3 = g1.to(device), g2.to(device), g3.to(device)
            labels = labels.to(device)

            # DataParallel 适配
            if isinstance(model, nn.DataParallel):
                preds = model.module(g1, g2, g3, False)
            else:
                preds = model(g1, g2, g3, False)
            
            # KDD Cup Label 在第 4 列
            target = labels[:, 3] if labels.dim() > 1 else labels
            
            val_labels.extend(target.cpu().tolist())
            val_preds.extend(preds.cpu().tolist())
    
    try:
        val_auc = roc_auc_score(val_labels, val_preds)
    except ValueError:
        val_auc = 0.0
    
    val_binary_preds = [1 if p >= 0.5 else 0 for p in val_preds]
    val_binary_labels = [int(l) for l in val_labels]
    
    val_acc = accuracy_score(val_binary_labels, val_binary_preds)
    val_f1 = f1_score(val_binary_labels, val_binary_preds, zero_division=0)
    
    report = classification_report(
        val_binary_labels, 
        val_binary_preds, 
        target_names=['Not Dropout (0)', 'Dropout (1)'], 
        zero_division=0,
        digits=4 
    )
    
    return val_auc, val_acc, val_f1, report

# =================================================================
# *** 2. t-SNE 可视化 (适配 KDD Cup 数据流) ***
# =================================================================
def visualize_tsne(model, test_dataloader, device, logger, save_path):
    logger.info("Starting t-SNE visualization...")
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Generating t-SNE"):
        with torch.no_grad():
            g1, g2, g3, labels = batch
            if g1 is None: continue
            
            g1, g2, g3 = g1.to(device), g2.to(device), g3.to(device)
            
            module = model.module if isinstance(model, nn.DataParallel) else model
            _, embeds = module(g1, g2, g3, False, return_embed=True)
            
            target = labels[:, 3]
            all_embeddings.append(embeds.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    if not all_embeddings: return

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 下采样
    if all_embeddings.shape[0] > 3000:
        indices = np.random.choice(all_embeddings.shape[0], 3000, replace=False)
        all_embeddings, all_labels = all_embeddings[indices], all_labels[indices]

    tsne_results = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0, n_jobs=-1).fit_transform(all_embeddings)
    
    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0], 
        'tsne-2': tsne_results[:, 1], 
        'Status': [('Dropout' if l==1 else 'Not Dropout') for l in all_labels]
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="tsne-1", y="tsne-2", hue="Status", palette={"Not Dropout": "#1f77b4", "Dropout": "#d62728"}, data=df, alpha=0.6)
    plt.title('t-SNE Visualization of Student Embeddings')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =================================================================
# *** 3. 门控分布提琴图 (适配 Homogeneous Graph) ***
# =================================================================
def visualize_gate_distribution(model, test_dataloader, device, logger, save_path):
    logger.info("Starting Gate visualization...")
    model.eval()
    all_gates, all_labels = [], []

    for batch in tqdm(test_dataloader, desc="Generating gate values"):
        with torch.no_grad():
            g1, g2, g3, labels = batch
            if g1 is None: continue
            
            g1, g2, g3 = g1.to(device), g2.to(device), g3.to(device)
            target = labels[:, 3]

            module = model.module if isinstance(model, nn.DataParallel) else model
            _, gates = module(g1, g2, g3, False, return_gate_values=True)
            
            if gates is not None:
                # KDD Cup: Mask is in ndata['target'] (1=Enrollment)
                mask = (g3.ndata['target'] == 1)
                if mask.sum() == 0: continue
                
                # Filter valid gates
                target_gates = gates[mask]
                
                # Safety crop for batch size mismatch
                limit = min(len(target_gates), len(target))
                
                all_gates.append(target_gates[:limit].cpu().numpy())
                all_labels.append(target[:limit].cpu().numpy())

    if len(all_gates) == 0: return

    # Flatten
    all_gates = np.mean(np.concatenate(all_gates, axis=0), axis=1) # [N_samples]
    all_labels = np.concatenate(all_labels, axis=0).astype(int)
    
    df = pd.DataFrame({'Gate Value (Avg)': all_gates, 'Status': [('Dropout' if l==1 else 'Not Dropout') for l in all_labels]})

    plt.figure(figsize=(10, 7))
    sns.violinplot(x="Status", y="Gate Value (Avg)", hue="Status", palette={"Not Dropout": "#1f77b4", "Dropout": "#d62728"}, data=df, legend=False)
    plt.title('Distribution of Adaptive Gate ($g_t$)')
    plt.ylabel('Gate Value ($g_t$)\n(0 = Long-Term, 1 = Short-Term)')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =================================================================
# *** [新增] 特征相关性分析 (针对 Reviewer 3) ***
# =================================================================
def analyze_correlation(model, dataloader, device, logger, save_path):
    logger.info("Starting Feature Correlation Analysis...")
    model.eval()
    all_gates = []
    all_raw_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            g1, g2, g3, _ = batch
            if g1 is None: continue
            g1, g2, g3 = g1.to(device), g2.to(device), g3.to(device)
            
            module = model.module if isinstance(model, nn.DataParallel) else model
            _, gate_val = module(g1, g2, g3, False, return_gate_values=True)
            
            mask = (g3.ndata['target'] == 1)
            if mask.sum() == 0 or gate_val is None: continue
            
            target_gates = gate_val[mask]
            target_feats = g3.ndata['feature'][mask]
            
            # Reduce to mean gate per student
            all_gates.extend(target_gates.mean(dim=1).cpu().numpy())
            all_raw_features.extend(target_feats.cpu().numpy())

    if not all_raw_features: return

    # Feature Mapping for KDD Cup
    raw_dim = len(all_raw_features[0])
    feature_names = []
    for i in range(raw_dim):
        if i < 3: feature_names.append(f"Type_Idx_{i}")
        elif i < raw_dim - 7: feature_names.append(f"Day_{i-3}_Activity") # Daily clicks
        else: feature_names.append(f"Course_Feat_{i}")

    df = pd.DataFrame(all_raw_features, columns=feature_names)
    df['Gate'] = all_gates
    
    # Calculate Correlation for Day Activity
    corr_dict = {}
    for feat in [f for f in feature_names if f.startswith("Day_")]:
        if np.std(df[feat]) > 1e-6:
            corr_dict[feat] = pearsonr(df['Gate'], df[feat])[0]
            
    if not corr_dict: return

    # Plot
    sorted_items = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    feats, corrs = zip(*sorted_items)
    
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in corrs]
    plt.barh(feats, corrs, color=colors)
    plt.title('Correlation between Daily Activity and Short-term Focus ($g_t$)')
    plt.xlabel('Pearson Correlation')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Correlation plot saved to {save_path}")

# =================================================================
# *** 4. 主训练函数 (集成所有修复与分析) ***
# =================================================================
def train(args, dataset, logger):
    # 1. 种子设置
    dgl.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 2. 数据划分 (Fix: Start form index 1)
    total_samples = len(dataset)
    indices = list(range(1, total_samples)) # KDD Cup Index 0 is full graph
    if not indices: return 

    np.random.shuffle(indices)
    split = int(np.floor(0.8 * len(indices)))
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    logger.info(f"Split: Train {len(train_indices)}, Test {len(test_indices)}")
    
    # 3. DataLoader
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                                  sampler=train_sampler, collate_fn=custom_collate, 
                                  num_workers=args.num_workers)
    
    # Test loader for eval (can use workers)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                                 sampler=test_sampler, collate_fn=custom_collate, 
                                 num_workers=args.num_workers)

    # 4. 设备 & 维度检测 (Fix: Homogeneous)
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
    
    try:
        g_sample = dataset[train_indices[0]][0] # Safe sample
        feat_dim = g_sample.ndata['feature'].shape[1]
        logger.info(f"Feature Dim detected: {feat_dim}")
    except:
        feat_dim = 20 # Fallback
    
    # 5. 模型初始化 (Fix: Smart Params)
    model_class = getattr(import_module('model'), args.model)
    
    # 检查是否需要异构参数
    sig = inspect.signature(model_class.__init__)
    init_args = {
        'num_layers': args.num_layers, 'h_feats': args.hidden_dim, 
        'bi': args.bi, 'dropout': args.dropout, 
        'batch_size': args.batch_size, 'device': device
    }
    
    if 'enroll_in_feats' in sig.parameters:
        init_args.update({'enroll_in_feats': feat_dim, 'object_in_feats': feat_dim, 'course_in_feats': feat_dim})
    else:
        init_args['in_feats'] = feat_dim

    model = model_class(**init_args).to(device)

    if len(args.gpu) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu)
    
    # === [意见2修改] 引入权重正则化 ===
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss().to(device)
    
    # 6. 训练循环 (含稳定性追踪)
    train_losses = []
    gate_stability_history = [] 

    logger.info(f"Training Start - Model: {args.model}, Seed: {args.seed}")

    for e in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.
        epoch_gates = []
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {e}/{args.num_epochs}"):
            g1, g2, g3, labels = batch
            if g1 is None: continue
            
            g1, g2, g3 = g1.to(device), g2.to(device), g3.to(device)
            labels = labels.to(device)
            target = labels[:, 3]

            # 提取门控值用于稳定性分析
            module = model.module if isinstance(model, nn.DataParallel) else model
            preds, gates = module(g1, g2, g3, True, return_gate_values=True)
            
            loss = loss_fn(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * preds.shape[0]
            
            if gates is not None:
                epoch_gates.append(gates.mean().item()) # 记录 batch 均值

        avg_loss = epoch_loss / len(train_indices)
        avg_gate = np.mean(epoch_gates) if epoch_gates else 0.0
        
        train_losses.append(avg_loss)
        gate_stability_history.append(avg_gate)

        logger.info(f"Epoch {e} - Loss: {avg_loss:.4f} - Stability g_t: {avg_gate:.4f}")

    # 7. 绘制稳定性分析图 (Reviewer 2)
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(range(1, args.num_epochs + 1), train_losses, 'b-', label='Loss')
    ax2.plot(range(1, args.num_epochs + 1), gate_stability_history, 'r--', label='Gate $g_t$')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('BCE Loss', color='b')
    ax2.set_ylabel('Avg Gate Value', color='r')
    plt.title('Convergence & Gating Stability')
    plt.savefig(f"{args.log_dir}/stability_plot.png")
    plt.close()

    # 8. 评估与可视化
    auc, acc, f1, report = eval(model, test_dataloader, device)
    logger.info(f"\nFinal Test AUC: {auc:.4f}, F1: {f1:.4f}\n{report}")

    # t-SNE & Gate Dist
    visualize_tsne(model, test_dataloader, device, logger, f"{args.log_dir}/tsne_plot.png")
    visualize_gate_distribution(model, test_dataloader, device, logger, f"{args.log_dir}/gate_plot.png")
    
    # 相关性分析 (Fix: Num_workers=0 to avoid CUDA crash)
    logger.info("Running Correlation Analysis (Safe Loader)...")
    analysis_loader = DataLoader(dataset, batch_size=args.batch_size, 
                                 sampler=test_sampler, collate_fn=custom_collate, 
                                 num_workers=0) # 强制单进程
    analyze_correlation(model, analysis_loader, device, logger, f"{args.log_dir}/correlation_plot.png")

    logger.info(f"Stability Sequence: {gate_stability_history}")

