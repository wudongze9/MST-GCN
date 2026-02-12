import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import dgl
from torch.optim import Adam
from model import *
from utils import collate_data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import pearsonr


def eval(model, test_dataloader, device):
    model.eval()
    val_labels = []
    val_preds = []
    
    for iter_idx, batch in tqdm(enumerate(test_dataloader, start=1), total=len(test_dataloader), desc="Evaluating"):
        with torch.no_grad():
            one_g = batch[0].to(device)
            two_g = batch[1].to(device)
            third_g = batch[2].to(device)
            labels = batch[3].to(device)

            if isinstance(model, nn.DataParallel):
                preds = model.module(one_g, two_g, third_g, False)
            else:
                preds = model(one_g, two_g, third_g, False)
            
        target = labels[:, 3] if labels.dim() > 1 else labels
        val_labels.extend(target.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())
    
    try:
        val_auc = roc_auc_score(val_labels, val_preds)
    except ValueError:
        val_auc = 0.0
    
    val_binary_preds = [round(p) for p in val_preds]
    val_binary_labels = [round(l) for l in val_labels]
    
    report = classification_report(
        val_binary_labels, 
        val_binary_preds, 
        target_names=['Not Dropout (0)', 'Dropout (1)'], 
        zero_division=0, digits=4 
    )
    return val_auc, f1_score(val_binary_labels, val_binary_preds, zero_division=0), report


def visualize_gate_distribution(model, test_dataloader, device, logger, save_path):
    logger.info("Starting Gate visualization...")
    model.eval()
    all_gate_means = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Generating gate values"):
        with torch.no_grad():
            one_g, two_g, third_g, labels = [b.to(device) if hasattr(b, 'to') else b for b in batch]
            
            if isinstance(model, nn.DataParallel):
                _, gates = model.module(one_g, two_g, third_g, False, return_gate_values=True)
            else:
                _, gates = model(one_g, two_g, third_g, False, return_gate_values=True)
            
            if gates is not None:
                target_mask = (third_g.nodes['enrollment'].data['target'] == 1)
                target_gates_avg = gates[target_mask].mean(dim=1).cpu().numpy()
                all_gate_means.append(target_gates_avg)
                
                target = labels[:, 3] if labels.dim() > 1 else labels
                all_labels.append(target.cpu().numpy())

    if not all_gate_means: return

    final_gates = np.concatenate(all_gate_means, axis=0)
    final_labels = np.concatenate(all_labels, axis=0).astype(int)
    
    df = pd.DataFrame({
        'Gate Value (Avg)': final_gates, 
        'Status': [('Dropout' if l==1 else 'Not Dropout') for l in final_labels]
    })

    plt.figure(figsize=(10, 7), dpi=300)
    sns.violinplot(x="Status", y="Gate Value (Avg)", hue="Status", 
                   palette={"Not Dropout": "#1f77b4", "Dropout": "#d62728"}, data=df, legend=False)
    plt.title('Distribution of Adaptive Gate ($g_t$)')
    plt.ylabel('Gate Value ($g_t$)\n(0 = Long-Term, 1 = Short-Term)')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_activity_gate_correlation(model, test_dataloader, device, logger, save_path):
    """
    分析日活跃度与门控值的相关性，并生成横向排序条形图。
    """
    logger.info("Starting Activity-Gate Correlation analysis...")
    model.eval()
    all_acts, all_gs = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            g3 = batch[2].to(device)
            if isinstance(model, nn.DataParallel):
                _, gates = model.module(batch[0].to(device), batch[1].to(device), g3, False, return_gate_values=True)
            else:
                _, gates = model(batch[0].to(device), batch[1].to(device), g3, False, return_gate_values=True)

            if gates is not None:
                target_mask = (g3.nodes['enrollment'].data['target'] == 1)
                # 假设 XuetangX 窗口为 10 天，特征从索引 3 开始
                feats = g3.nodes['enrollment'].data['feature'][target_mask].cpu().numpy()
                daily_act = feats[:, 3:13] 
                target_gates = gates[target_mask].mean(dim=1).cpu().numpy()
                
                all_acts.append(daily_act)
                all_gs.append(target_gates)

    if not all_acts: return
    
    X = np.concatenate(all_acts, axis=0) # [N_samples, 10]
    Y = np.concatenate(all_gs, axis=0)   # [N_samples]
    
    # 计算每一天的 Pearson 相关系数
    corrs = []
    days = [f'Day_{i}_Activity' for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        c, _ = pearsonr(X[:, i], Y)
        corrs.append(c)
    
    # 构造 DataFrame 并按相关系数降序排列
    corr_df = pd.DataFrame({'Day': days, 'Correlation': corrs})
    corr_df = corr_df.sort_values(by='Correlation', ascending=False)

    # 绘图逻辑：对齐你提供的横向样式
    plt.figure(figsize=(12, 8), dpi=300)
    # 正相关用红色，负相关用蓝色
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in corr_df['Correlation']]
    
    sns.barplot(x='Correlation', y='Day', data=corr_df, palette=colors)
    
    plt.title('Correlation between Daily Activity and Short-term Focus ($g_t$)', fontsize=14)
    plt.xlabel('Pearson Correlation', fontsize=12)
    plt.ylabel('', fontsize=12) # 隐藏 Y 轴标签（Days 已在刻度显示）
    plt.axvline(0, color='black', linewidth=0.8) # 在 0 处画分割线
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_tsne(model, test_dataloader, device, logger, save_path):
    model.eval()
    all_embeds, all_labels = [], []
    for batch in test_dataloader:
        with torch.no_grad():
            g1, g2, g3, labels = [b.to(device) for b in batch]
            if isinstance(model, nn.DataParallel):
                _, embeds = model.module(g1, g2, g3, False, return_embed=True)
            else:
                _, embeds = model(g1, g2, g3, False, return_embed=True)
            all_embeds.append(embeds.cpu().numpy())
            all_labels.append((labels[:, 3] if labels.dim() > 1 else labels).cpu().numpy())

    X_emb = np.concatenate(all_embeds, axis=0)
    Y_lab = np.concatenate(all_labels, axis=0)
    if X_emb.shape[0] > 3000:
        idx = np.random.choice(X_emb.shape[0], 3000, replace=False)
        X_emb, Y_lab = X_emb[idx], Y_lab[idx]

    res = TSNE(n_components=2, random_state=0).fit_transform(X_emb)
    df = pd.DataFrame({'tsne-1': res[:, 0], 'tsne-2': res[:, 1], 'Status': [('Dropout' if l==1 else 'Not Dropout') for l in Y_lab]})
    plt.figure(figsize=(10, 7), dpi=300)
    sns.scatterplot(x="tsne-1", y="tsne-2", hue="Status", palette={"Not Dropout": "#1f77b4", "Dropout": "#d62728"}, data=df, alpha=0.6)
    plt.title('t-SNE Visualization of Student Embeddings')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train(args, dataloader, logger):
    dgl.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
    
    sample = dataloader[0]
    g_sample = sample[0]
    in_dim = {ntype: g_sample.nodes[ntype].data['feature'].shape[1] for ntype in g_sample.ntypes}

    model = Multi_MST_GCN(num_layers=args.num_layers, enroll_in_feats=in_dim['enrollment'], 
                          object_in_feats=in_dim['object'], course_in_feats=in_dim['course'],
                          h_feats=args.hidden_dim, bi=args.bi, dropout=args.dropout, 
                          batch_size=args.batch_size, device=device).to(device)

    if len(args.gpu) > 1: model = nn.DataParallel(model, device_ids=args.gpu)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss().to(device)
    
    indices = list(range(len(dataloader)))
    split = int(len(dataloader) * 0.8)
    train_dl = DataLoader(dataloader, collate_fn=collate_data, sampler=SubsetRandomSampler(indices[:split]), batch_size=args.batch_size)
    test_dl = DataLoader(dataloader, collate_fn=collate_data, sampler=SubsetRandomSampler(indices[split:]), batch_size=args.batch_size)
    
    losses, gate_stability = [], []

    for e in range(1, args.num_epochs + 1):
        model.train()
        epoch_l, epoch_g = 0., []
        for batch in tqdm(train_dl, desc=f"Epoch {e}"):
            g1, g2, g3, labels = [b.to(device) for b in batch]
            preds, gates = model(g1, g2, g3, True, return_gate_values=True)
            loss = loss_fn(preds, labels[:, 3] if labels.dim() > 1 else labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_l += loss.item() * preds.shape[0]
            if gates is not None: epoch_g.append(gates.mean().item())

        avg_l, avg_g = epoch_l / len(indices[:split]), np.mean(epoch_g)
        losses.append(avg_l); gate_stability.append(avg_g)
        logger.info(f"Epoch {e}: Loss {avg_l:.4f}, Gate {avg_g:.4f}")

    # 稳定性监控图
    plt.figure(figsize=(10, 6), dpi=300)
    ax1 = plt.gca(); ax2 = ax1.twinx()
    ax1.plot(range(1, args.num_epochs+1), losses, 'b-', label='Loss')
    ax2.plot(range(1, args.num_epochs+1), gate_stability, 'r--', label='Gate $g_t$')
    plt.savefig(f"{args.log_dir}/stability_plot.png")
    plt.close()

    # 执行评估与所有可视化
    auc, f1, report = eval(model, test_dl, device)
    logger.info(f"Final AUC: {auc:.4f}, F1: {f1:.4f}\n{report}")
    
    visualize_tsne(model, test_dl, device, logger, f"{args.log_dir}/tsne_plot.png")
    visualize_gate_distribution(model, test_dl, device, logger, f"{args.log_dir}/gate_plot.png")
    visualize_activity_gate_correlation(model, test_dl, device, logger, f"{args.log_dir}/correlation_plot.png")