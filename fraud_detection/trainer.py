"""
训练模块 - 课程学习 (二分类 + 多分类)
"""
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

import config
from model import FraudDetectionModel, MultiTaskModel
from dataset import load_and_preprocess_data, prepare_curriculum_dataloaders
from transformers import BertTokenizer


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============== 二分类训练函数 ==============

def train_binary_epoch(model, dataloader, optimizer, scheduler, device):
    """训练二分类一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training Binary"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['is_fraud_label'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate_binary(model, dataloader, device):
    """评估二分类模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['is_fraud_label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    report = classification_report(all_labels, all_preds, labels=[0, 1], 
                                   target_names=['非诈骗', '诈骗'], output_dict=True, zero_division=0)
    
    # 计算 weighted avg
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': avg_loss, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'report': report,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


def print_binary_stage_result(metrics, stage_name=""):
    """打印二分类阶段结果"""
    print(f" ")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    


def print_binary_final_result(metrics):
    """打印二分类最终详细结果"""
    print(f"\n{'='*60}")
    print(f"二分类 最终评估结果")
    print(f"{'='*60}")
    print(f"\n总体结果:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision_weighted']:.4f}")
    print(f"Recall:     {metrics['recall_weighted']:.4f}")
    print(f"F1-score:   {metrics['f1_weighted']:.4f}")
    
    print(f"\n具体类别结果:")
    print(f"{'-'*60}")
    print(f"{'':10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print(f"{'-'*60}")
    for cls in ['非诈骗', '诈骗']:
        if cls in metrics['report']:
            r = metrics['report'][cls]
            print(f"{cls:10} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")
    print(f"{'-'*60}")


# ============== 多分类训练函数 ==============

def train_fraud_type_epoch(model, dataloader, optimizer, scheduler, device, num_classes=7):
    """训练多分类一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training FraudType"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['fraud_type_label'].to(device)
        
        _, fraud_type_logits = model(input_ids, attention_mask)
        loss = loss_fn(fraud_type_logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(fraud_type_logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate_fraud_type(model, dataloader, device, num_classes=7):
    """评估多分类模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['fraud_type_label'].to(device)
            
            _, fraud_type_logits = model(input_ids, attention_mask)
            loss = loss_fn(fraud_type_logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(fraud_type_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    report = classification_report(all_labels, all_preds, 
                                   labels=list(range(num_classes)),
                                   target_names=config.FRAUD_TYPES, 
                                   output_dict=True, zero_division=0)
    
    # 计算 macro avg 和 weighted avg
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': avg_loss, 
        'accuracy': accuracy, 
        'precision_macro': precision_m,
        'recall_macro': recall_m,
        'f1_macro': f1_m,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'report': report,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


def print_fraud_type_stage_result(metrics, stage_name=""):
    """打印多分类阶段结果"""
    print(f" ")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall:    {metrics['recall_weighted']:.4f}")
    print(f"F1-score:  {metrics['f1_weighted']:.4f}")


def evaluate_fraud_type_by_difficulty(model, eval_df, tokenizer, device, batch_size=16):
    """
    按难度评估诈骗类型多分类模型（验证集）
    返回每个难度的评估指标
    """
    model.eval()
    results = {}
    
    for difficulty in ['easy', 'medium', 'hard']:
        diff_df = eval_df[eval_df['difficulty'] == difficulty].reset_index(drop=True)
        
        if len(diff_df) == 0:
            continue
        
        dataset = FraudDataset(diff_df, tokenizer, for_fraud_type=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['fraud_type_label'].to(device)
                
                _, fraud_type_logits = model(input_ids, attention_mask)
                preds = torch.argmax(fraud_type_logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        if len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            
            results[difficulty] = {
                'accuracy': accuracy,
                'precision': precision_w,
                'recall': recall_w,
                'f1_score': f1_w,
                'support': len(all_labels)
            }
    
    return results
    


def print_fraud_type_final_result(metrics, difficulty_metrics=None):
    """打印多分类最终详细结果"""
    print(f"\n{'='*70}")
    print(f"多分类 最终评估结果 (验证集)")
    print(f"{'='*70}")
    print(f"\n总体结果:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision_weighted']:.4f}")
    print(f"Recall:     {metrics['recall_weighted']:.4f}")
    print(f"F1-score:   {metrics['f1_weighted']:.4f}")
    
    # 按难度输出结果
    if difficulty_metrics:
        print(f"\n按难度分类结果:")
        print(f"{'-'*60}")
        print(f"{'难度':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
        print(f"{'-'*60}")
        difficulty_names = {'easy': '简单', 'medium': '中等', 'hard': '困难'}
        for diff in ['easy', 'medium', 'hard']:
            if diff in difficulty_metrics:
                m = difficulty_metrics[diff]
                print(f"{difficulty_names[diff]:>10} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['support']:>10}")
        print(f"{'-'*60}")
    
    print(f"\n具体类别结果:")
    print(f"{'-'*70}")
    print(f"{'':12} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Support':>10}")
    print(f"{'-'*70}")
    for fraud_type in config.FRAUD_TYPES:
        if fraud_type in metrics['report']:
            r = metrics['report'][fraud_type]
            print(f"{fraud_type:12} {r['precision']:>12.4f} {r['recall']:>12.4f} {r['f1-score']:>12.4f} {int(r['support']):>10}")
    
    # 打印 macro avg
    if 'macro avg' in metrics['report']:
        ma = metrics['report']['macro avg']
        print(f"{'-'*70}")
        print(f"{'macro avg':12} {ma['precision']:>12.4f} {ma['recall']:>12.4f} {ma['f1-score']:>12.4f} {int(ma['support']):>10}")
    print(f"{'='*70}")


# ============== 二分类课程学习训练 ==============

def train_curriculum_learning_binary():
    """
    二分类课程学习训练
    返回: history (包含loss曲线数据)
    """
    set_seed(42)
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[二分类] 使用设备: {device}")
    
    # 加载数据
    print("\n[二分类] 加载训练数据...")
    df = load_and_preprocess_data(config.TRAIN_FILE)
    
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG['model_name'])
    
    print("[二分类] 准备课程学习数据加载器...")
    dataloaders = prepare_curriculum_dataloaders(
        df, tokenizer,
        batch_size=config.MODEL_CONFIG['batch_size'],
        eval_ratio=config.MODEL_CONFIG['eval_ratio']
    )
    
    # 记录训练历史
    history = {
        'stage1': {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []},
        'stage2': {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []},
        'stage3': {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []},
    }
    
    model = MultiTaskModel(model_name=config.MODEL_CONFIG['model_name'])
    model.to(device)
    
    # 阶段1: Easy
    print("\n" + "="*60)
    print("二分类 - 阶段1: Easy样本 (7轮及以内)")
    print("="*60)
    
    train_loader = dataloaders['easy']['train_loader']
    eval_loader = dataloaders['easy']['eval_loader']
    num_epochs = config.BINARY_CONFIG['num_epochs_stage1']
    num_steps = len(train_loader) * num_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.MODEL_CONFIG['learning_rate'], 
                       weight_decay=config.MODEL_CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(num_steps * 0.1),
                                                num_training_steps=num_steps)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_binary_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_binary(model, eval_loader, device)
        
        history['stage1']['train_loss'].append(train_loss)
        history['stage1']['eval_loss'].append(eval_metrics['loss'])
        history['stage1']['train_acc'].append(train_acc)
        history['stage1']['eval_acc'].append(eval_metrics['accuracy'])
        
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  评估 Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
    
    # 阶段1结束，评估并打印结果
    stage1_metrics = evaluate_binary(model, eval_loader, device)
    print_binary_stage_result(stage1_metrics, "阶段1")
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'binary_stage1.pt'))
    
    # 阶段2: Easy + Medium
    print("\n" + "="*60)
    print("二分类 - 阶段2: Easy + Medium样本 (8-15轮)")
    print("="*60)
    
    train_loader = dataloaders['stage2']['train_loader']
    eval_loader = dataloaders['stage2']['eval_loader']
    num_epochs = config.BINARY_CONFIG['num_epochs_stage2']
    num_steps = len(train_loader) * num_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.MODEL_CONFIG['learning_rate'] * 0.5,
                      weight_decay=config.MODEL_CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_steps * 0.1),
                                                num_training_steps=num_steps)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_binary_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_binary(model, eval_loader, device)
        
        history['stage2']['train_loss'].append(train_loss)
        history['stage2']['eval_loss'].append(eval_metrics['loss'])
        history['stage2']['train_acc'].append(train_acc)
        history['stage2']['eval_acc'].append(eval_metrics['accuracy'])
        
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  评估 Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
    
    # 阶段2结束，评估并打印结果
    stage2_metrics = evaluate_binary(model, eval_loader, device)
    print_binary_stage_result(stage2_metrics, "阶段2")
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'binary_stage2.pt'))
    
    # 阶段3: All
    print("\n" + "="*60)
    print("二分类 - 阶段3: 全部样本")
    print("="*60)
    
    train_loader = dataloaders['stage3']['train_loader']
    eval_loader = dataloaders['stage3']['eval_loader']
    num_epochs = config.BINARY_CONFIG['num_epochs_stage3']
    num_steps = len(train_loader) * num_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.MODEL_CONFIG['learning_rate'] * 0.25,
                      weight_decay=config.MODEL_CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_steps * 0.1),
                                                num_training_steps=num_steps)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_binary_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_binary(model, eval_loader, device)
        
        history['stage3']['train_loss'].append(train_loss)
        history['stage3']['eval_loss'].append(eval_metrics['loss'])
        history['stage3']['train_acc'].append(train_acc)
        history['stage3']['eval_acc'].append(eval_metrics['accuracy'])
        
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  评估 Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
    
    # 阶段3结束，评估并打印结果
    stage3_metrics = evaluate_binary(model, eval_loader, device)
    print_binary_stage_result(stage3_metrics, "阶段3")
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'binary_model.pt'))
    
    # 最终评估
    final_metrics = evaluate_binary(model, eval_loader, device)
    print_binary_final_result(final_metrics)
    
    # 保存各阶段指标到history
    history['stage_metrics'] = {
        'stage1': {'precision': stage1_metrics['precision'], 'recall': stage1_metrics['recall'], 'f1': stage1_metrics['f1']},
        'stage2': {'precision': stage2_metrics['precision'], 'recall': stage2_metrics['recall'], 'f1': stage2_metrics['f1']},
        'stage3': {'precision': stage3_metrics['precision'], 'recall': stage3_metrics['recall'], 'f1': stage3_metrics['f1']},
    }
    
    return model, history, final_metrics, difficulty_metrics


# ============== 多分类课程学习训练 ==============

def train_curriculum_learning_fraud_type():
    """
    多分类课程学习训练 (从头开始，不依赖二分类模型)
    返回: history (包含loss曲线数据)
    """
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[多分类] 使用设备: {device}")
    
    # 加载数据 (仅诈骗样本)
    print("\n[多分类] 加载训练数据...")
    df = load_and_preprocess_data(config.TRAIN_FILE)
    fraud_df = df[df['is_fraud_binary'] == 1].reset_index(drop=True)
    print(f"[多分类] 诈骗样本数量: {len(fraud_df)}")
    
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG['model_name'])
    
    print("[多分类] 准备课程学习数据加载器...")
    dataloaders = prepare_curriculum_dataloaders(
        fraud_df, tokenizer,
        batch_size=config.MODEL_CONFIG['batch_size'],
        eval_ratio=config.MODEL_CONFIG['eval_ratio'],
        for_fraud_type=True
    )
    
    # 记录训练历史
    history = {
        'stage1': {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []},
        'stage2': {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []},
        'stage3': {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []},
    }
    
    # 创建新模型 (从头开始)
    model = FraudDetectionModel(num_fraud_types=len(config.FRAUD_TYPES))
    model.to(device)
    
    # 阶段1: Easy
    print("\n" + "="*60)
    print("多分类 - 阶段1: Easy样本")
    print("="*60)
    
    train_loader = dataloaders['easy']['train_loader']
    eval_loader = dataloaders['easy']['eval_loader']
    num_epochs = config.FRAUD_TYPE_CONFIG['num_epochs_stage1']
    num_steps = len(train_loader) * num_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.MODEL_CONFIG['learning_rate'],
                      weight_decay=config.MODEL_CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_steps * 0.1),
                                                num_training_steps=num_steps)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_fraud_type_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_fraud_type(model, eval_loader, device)
        
        history['stage1']['train_loss'].append(train_loss)
        history['stage1']['eval_loss'].append(eval_metrics['loss'])
        history['stage1']['train_acc'].append(train_acc)
        history['stage1']['eval_acc'].append(eval_metrics['accuracy'])
        
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  评估 Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
    
    # 阶段1结束，评估并打印结果
    stage1_metrics = evaluate_fraud_type(model, eval_loader, device)
    print_fraud_type_stage_result(stage1_metrics, "阶段1")
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'fraud_type_stage1.pt'))
    
    # 阶段2: Easy + Medium
    print("\n" + "="*60)
    print("多分类 - 阶段2: Easy + Medium样本")
    print("="*60)
    
    train_loader = dataloaders['stage2']['train_loader']
    eval_loader = dataloaders['stage2']['eval_loader']
    num_epochs = config.FRAUD_TYPE_CONFIG['num_epochs_stage2']
    num_steps = len(train_loader) * num_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.MODEL_CONFIG['learning_rate'] * 0.5,
                      weight_decay=config.MODEL_CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_steps * 0.1),
                                                num_training_steps=num_steps)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_fraud_type_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_fraud_type(model, eval_loader, device)
        
        history['stage2']['train_loss'].append(train_loss)
        history['stage2']['eval_loss'].append(eval_metrics['loss'])
        history['stage2']['train_acc'].append(train_acc)
        history['stage2']['eval_acc'].append(eval_metrics['accuracy'])
        
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  评估 Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
    
    # 阶段2结束，评估并打印结果
    stage2_metrics = evaluate_fraud_type(model, eval_loader, device)
    print_fraud_type_stage_result(stage2_metrics, "阶段2")
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'fraud_type_stage2.pt'))
    
    # 阶段3: All
    print("\n" + "="*60)
    print("多分类 - 阶段3: 全部样本")
    print("="*60)
    
    train_loader = dataloaders['stage3']['train_loader']
    eval_loader = dataloaders['stage3']['eval_loader']
    num_epochs = config.FRAUD_TYPE_CONFIG['num_epochs_stage3']
    num_steps = len(train_loader) * num_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.MODEL_CONFIG['learning_rate'] * 0.25,
                      weight_decay=config.MODEL_CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_steps * 0.1),
                                                num_training_steps=num_steps)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_fraud_type_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = evaluate_fraud_type(model, eval_loader, device)
        
        history['stage3']['train_loss'].append(train_loss)
        history['stage3']['eval_loss'].append(eval_metrics['loss'])
        history['stage3']['train_acc'].append(train_acc)
        history['stage3']['eval_acc'].append(eval_metrics['accuracy'])
        
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  评估 Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
    
    # 阶段3结束，评估并打印结果
    stage3_metrics = evaluate_fraud_type(model, eval_loader, device)
    print_fraud_type_stage_result(stage3_metrics, "阶段3")
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'fraud_type_model.pt'))
    
    # 最终评估
    final_metrics = evaluate_fraud_type(model, eval_loader, device)
    
    # 按难度评估（使用stage3的eval_df）
    stage3_eval_df = dataloaders['stage3']['eval_df']
    difficulty_metrics = evaluate_fraud_type_by_difficulty(
        model, stage3_eval_df, tokenizer, device, 
        batch_size=config.MODEL_CONFIG['batch_size']
    )
    
    print_fraud_type_final_result(final_metrics, difficulty_metrics)
    
    # 保存各阶段指标到history
    history['stage_metrics'] = {
        'stage1': {'precision': stage1_metrics['precision_weighted'], 'recall': stage1_metrics['recall_weighted'], 'f1': stage1_metrics['f1_weighted']},
        'stage2': {'precision': stage2_metrics['precision_weighted'], 'recall': stage2_metrics['recall_weighted'], 'f1': stage2_metrics['f1_weighted']},
        'stage3': {'precision': stage3_metrics['precision_weighted'], 'recall': stage3_metrics['recall_weighted'], 'f1': stage3_metrics['f1_weighted']},
    }
    
    return model, history, final_metrics, difficulty_metrics


# ============== 绘图函数 ==============

def plot_training_history(history, title, save_path):
    """绘制训练loss曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    stages = ['stage1', 'stage2', 'stage3']
    stage_names = ['Stage1 (Easy)', 'Stage2 (Easy+Medium)', 'Stage3 (All)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Loss曲线
    epoch_offset = 0
    for i, (stage, name) in enumerate(zip(stages, stage_names)):
        epochs = list(range(epoch_offset, epoch_offset + len(history[stage]['train_loss'])))
        epoch_offset += len(history[stage]['train_loss'])
        
        ax1.plot(epochs, history[stage]['train_loss'], 'o-', color=colors[i], 
                 label=f'{name} Train', linewidth=2)
        ax1.plot(epochs, history[stage]['eval_loss'], 's--', color=colors[i], 
                 alpha=0.7, label=f'{name} Eval', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title} - Loss', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy曲线
    epoch_offset = 0
    for i, (stage, name) in enumerate(zip(stages, stage_names)):
        epochs = list(range(epoch_offset, epoch_offset + len(history[stage]['train_acc'])))
        epoch_offset += len(history[stage]['train_acc'])
        
        ax2.plot(epochs, history[stage]['train_acc'], 'o-', color=colors[i], 
                 label=f'{name} Train', linewidth=2)
        ax2.plot(epochs, history[stage]['eval_acc'], 's--', color=colors[i], 
                 alpha=0.7, label=f'{name} Eval', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{title} - Accuracy', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss曲线已保存: {save_path}")


def save_binary_report(history, final_metrics):
    """保存二分类训练报告"""
    # 构建各类别的详细指标
    per_class = {}
    for cls in ['非诈骗', '诈骗']:
        if cls in final_metrics['report']:
            r = final_metrics['report'][cls]
            per_class[cls] = {
                'precision': r['precision'],
                'recall': r['recall'],
                'f1-score': r['f1-score'],
                'support': r['support'],
            }
    
    report = {
        'task': '二分类 (Binary Classification)',
        'curriculum_learning': True,
        'epochs_per_stage': {
            'stage1': config.BINARY_CONFIG['num_epochs_stage1'],
            'stage2': config.BINARY_CONFIG['num_epochs_stage2'],
            'stage3': config.BINARY_CONFIG['num_epochs_stage3'],
        },
        'eval_ratio': config.MODEL_CONFIG['eval_ratio'],
        'stage_metrics': history.get('stage_metrics', {}),
        'final_metrics': {
            'total': {
                'accuracy': final_metrics['accuracy'],
                'precision': final_metrics['precision_weighted'],
                'recall': final_metrics['recall_weighted'],
                'f1-score': final_metrics['f1_weighted'],
            },
            'per_class': per_class
        }
    }
    
    report_path = os.path.join(config.REPORT_DIR, 'binary_train_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"二分类报告已保存: {report_path}")


def save_fraud_type_report(history, final_metrics):
    """保存多分类训练报告"""
    # 构建各类别的详细指标
    per_class = {}
    for fraud_type in config.FRAUD_TYPES:
        if fraud_type in final_metrics['report']:
            r = final_metrics['report'][fraud_type]
            per_class[fraud_type] = {
                'precision': r['precision'],
                'recall': r['recall'],
                'f1-score': r['f1-score'],
                'support': r['support'],
            }
    
    # macro avg
    macro_avg = {}
    if 'macro avg' in final_metrics['report']:
        ma = final_metrics['report']['macro avg']
        macro_avg = {
            'precision': ma['precision'],
            'recall': ma['recall'],
            'f1-score': ma['f1-score'],
            'support': ma['support'],
        }
    
    report = {
        'task': '多分类 (Fraud Type Classification)',
        'curriculum_learning': True,
        'epochs_per_stage': {
            'stage1': config.FRAUD_TYPE_CONFIG['num_epochs_stage1'],
            'stage2': config.FRAUD_TYPE_CONFIG['num_epochs_stage2'],
            'stage3': config.FRAUD_TYPE_CONFIG['num_epochs_stage3'],
        },
        'eval_ratio': config.MODEL_CONFIG['eval_ratio'],
        'stage_metrics': history.get('stage_metrics', {}),
        'final_metrics': {
            'total': {
                'accuracy': final_metrics['accuracy'],
                'precision': final_metrics['precision_weighted'],
                'recall': final_metrics['recall_weighted'],
                'f1-score': final_metrics['f1_weighted'],
            },
            'macro_avg': macro_avg,
            'per_class': per_class
        }
    }
    
    report_path = os.path.join(config.REPORT_DIR, 'fraud_type_train_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"多分类报告已保存: {report_path}")


# ============== 主训练函数 ==============

def train_all():
    """训练全部任务"""
    print("\n" + "="*60)
    print("开始完整训练流程")
    print("="*60)
    
    # 1. 二分类课程学习
    print("\n" + "="*60)
    print("阶段A: 二分类课程学习")
    print("="*60)
    binary_model, binary_history, binary_metrics = train_curriculum_learning_binary()
    
    # 保存二分类报告和曲线
    save_binary_report(binary_history, binary_metrics)
    plot_training_history(binary_history, 'Binary Classification', 
                          os.path.join(config.REPORT_DIR, 'binary_loss_curve.png'))
    
    # 2. 多分类课程学习
    print("\n" + "="*60)
    print("阶段B: 多分类课程学习")
    print("="*60)
    fraud_model, fraud_history, fraud_metrics = train_curriculum_learning_fraud_type()
    
    # 保存多分类报告和曲线
    save_fraud_type_report(fraud_history, fraud_metrics)
    plot_training_history(fraud_history, 'Fraud Type Classification', 
                         os.path.join(config.REPORT_DIR, 'fraud_type_loss_curve.png'))
    
    print("\n" + "="*60)
    print("所有训练完成!")
    print("="*60)
    print(f"模型保存目录: {config.MODEL_SAVE_DIR}")
    print(f"报告保存目录: {config.REPORT_DIR}")


def train_binary_only():
    """仅训练二分类"""
    print("\n" + "="*60)
    print("仅训练二分类任务")
    print("="*60)
    
    binary_model, binary_history, binary_metrics = train_curriculum_learning_binary()
    save_binary_report(binary_history, binary_metrics)
    plot_training_history(binary_history, 'Binary Classification',
                          os.path.join(config.REPORT_DIR, 'binary_loss_curve.png'))
    
    print("\n二分类训练完成!")


def train_fraud_type_only():
    """仅训练多分类"""
    print("\n" + "="*60)
    print("仅训练多分类任务")
    print("="*60)
    
    fraud_model, fraud_history, fraud_metrics = train_curriculum_learning_fraud_type()
    save_fraud_type_report(fraud_history, fraud_metrics)
    plot_training_history(fraud_history, 'Fraud Type Classification',
                         os.path.join(config.REPORT_DIR, 'fraud_type_loss_curve.png'))
    
    print("\n多分类训练完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练诈骗电话检测模型')
    parser.add_argument('task', type=str, nargs='?', default='all', 
                       choices=['all', '1', '2'],
                       help='训练任务: all(全部), 1(二分类), 2(多分类)')
    args = parser.parse_args()
    
    if args.task == 'all':
        train_all()
    elif args.task == '1':
        train_binary_only()
    elif args.task == '2':
        train_fraud_type_only()
