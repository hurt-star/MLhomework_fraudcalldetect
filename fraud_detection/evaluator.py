"""
评估模块
"""
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
from transformers import BertTokenizer

import config
from model import FraudDetectionModel, MultiTaskModel
from dataset import load_and_preprocess_data, FraudDataset


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_binary_classification(model, dataloader, device):
    """
    评估二分类模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['is_fraud_label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # 计算 weighted avg
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    report = classification_report(all_labels, all_preds, labels=[0, 1], target_names=['非诈骗', '诈骗'], output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, labels=[0, 1], target_names=['非诈骗', '诈骗'], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'classification_report': report,
        'classification_report_text': report_text,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, all_preds, all_labels


def evaluate_fraud_type(model, dataloader, device):
    """
    评估诈骗类型多分类模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估诈骗类型中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['fraud_type_label'].to(device)
            
            _, fraud_type_logits = model(input_ids, attention_mask)
            preds = torch.argmax(fraud_type_logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    labels_list = list(range(len(config.FRAUD_TYPES)))
    
    # 计算 macro avg 和 weighted avg
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    report = classification_report(all_labels, all_preds, labels=labels_list, target_names=config.FRAUD_TYPES, output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, labels=labels_list, target_names=config.FRAUD_TYPES, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_m,
        'recall_macro': recall_m,
        'f1_macro': f1_m,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'classification_report': report,
        'classification_report_text': report_text,
        'confusion_matrix': cm.tolist() if len(cm) <= len(config.FRAUD_TYPES) else None
    }
    
    return metrics, all_preds, all_labels


def evaluate_fraud_type_by_difficulty(model, fraud_df, tokenizer, device, batch_size=16):
    """
    按难度评估诈骗类型多分类模型
    返回每个难度的评估指标
    """
    model.eval()
    results = {}
    
    for difficulty in ['easy', 'medium', 'hard']:
        diff_df = fraud_df[fraud_df['difficulty'] == difficulty].reset_index(drop=True)
        
        if len(diff_df) == 0:
            continue
        
        dataset = FraudDataset(diff_df, tokenizer, for_fraud_type=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"评估{difficulty}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['fraud_type_label'].to(device)
                
                _, fraud_type_logits = model(input_ids, attention_mask)
                preds = torch.argmax(fraud_type_logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        if len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            
            results[difficulty] = {
                'accuracy': accuracy,
                'precision': precision_w,
                'recall': recall_w,
                'f1_score': f1_w,
                'precision_macro': precision_m,
                'recall_macro': recall_m,
                'f1_score_macro': f1_m,
                'support': len(all_labels)
            }
    
    return results


def evaluate_full_model(model, dataloader, device):
    """
    完整模型评估 (二分类 + 诈骗类型)
    """
    model.eval()
    all_is_fraud_preds = []
    all_is_fraud_labels = []
    all_fraud_type_preds = []
    all_fraud_type_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="完整评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            is_fraud_labels = batch['is_fraud_label'].to(device)
            
            is_fraud_logits, fraud_type_logits = model(input_ids, attention_mask)
            
            is_fraud_preds = torch.argmax(is_fraud_logits, dim=1).cpu().numpy()
            all_is_fraud_preds.extend(is_fraud_preds)
            all_is_fraud_labels.extend(is_fraud_labels.cpu().numpy())
            
            # 诈骗类型预测 (仅对预测为诈骗的样本)
            fraud_mask = is_fraud_preds == 1
            if fraud_mask.any():
                fraud_type_preds = torch.argmax(fraud_type_logits, dim=1).cpu().numpy()
                fraud_type_labels = batch['fraud_type_label'].numpy()
                all_fraud_type_preds.extend(fraud_type_preds[fraud_mask])
                all_fraud_type_labels.extend(fraud_type_labels[fraud_mask])
    
    # 二分类指标
    is_fraud_accuracy = accuracy_score(all_is_fraud_labels, all_is_fraud_preds)
    is_fraud_report_text = classification_report(all_is_fraud_labels, all_is_fraud_preds, 
                                                 labels=[0, 1], target_names=['非诈骗', '诈骗'], zero_division=0)
    
    # 诈骗类型指标 (如果有足够的诈骗样本)
    fraud_type_metrics = {}
    if len(all_fraud_type_preds) > 0:
        fraud_type_accuracy = accuracy_score(all_fraud_type_labels, all_fraud_type_preds)
        fraud_type_report_text = classification_report(
            all_fraud_type_labels, all_fraud_type_preds,
            labels=list(range(len(config.FRAUD_TYPES))),
            target_names=config.FRAUD_TYPES, zero_division=0
        )
        fraud_type_metrics = {
            'accuracy': fraud_type_accuracy,
            'report_text': fraud_type_report_text
        }
    
    metrics = {
        'is_fraud': {
            'accuracy': is_fraud_accuracy,
            'report_text': is_fraud_report_text
        },
        'fraud_type': fraud_type_metrics
    }
    
    return metrics


def print_binary_test_result(metrics):
    """打印二分类测试详细结果"""
    print(f"\n{'='*60}")
    print(f"二分类 测试集评估结果")
    print(f"{'='*60}")
    print(f"\n总体结果:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall:    {metrics['recall_weighted']:.4f}")
    print(f"F1-score:  {metrics['f1_weighted']:.4f}")
    
    print(f"\n具体类别结果:")
    print(f"{'-'*60}")
    print(f"{'':10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print(f"{'-'*60}")
    for cls in ['非诈骗', '诈骗']:
        if cls in metrics['classification_report']:
            r = metrics['classification_report'][cls]
            print(f"{cls:10} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")
    print(f"{'-'*60}")
    print(f"{'='*60}")


def print_fraud_type_test_result(metrics, difficulty_metrics=None):
    """打印多分类测试详细结果"""
    print(f"\n{'='*70}")
    print(f"多分类 测试集评估结果")
    print(f"{'='*70}")
    print(f"\n总体结果:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_weighted']:.4f}")
    print(f"Recall:    {metrics['recall_weighted']:.4f}")
    print(f"F1-score:  {metrics['f1_weighted']:.4f}")
    
    # 按难度输出结果
    if difficulty_metrics:
        print(f"\n不同难度分类结果:")
        print(f"{'-'*70}")
        print(f"{'难度':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
        print(f"{'-'*70}")
        difficulty_names = {'easy': '简单', 'medium': '中等', 'hard': '困难'}
        for diff in ['easy', 'medium', 'hard']:
            if diff in difficulty_metrics:
                m = difficulty_metrics[diff]
                print(f"{difficulty_names[diff]:>10} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['support']:>10}")
        print(f"{'-'*70}")
    
    print(f"\n具体类别结果:")
    print(f"{'-'*70}")
    print(f"{'':12} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Support':>10}")
    print(f"{'-'*70}")
    for fraud_type in config.FRAUD_TYPES:
        if fraud_type in metrics['classification_report']:
            r = metrics['classification_report'][fraud_type]
            print(f"{fraud_type:12} {r['precision']:>12.4f} {r['recall']:>12.4f} {r['f1-score']:>12.4f} {int(r['support']):>10}")
    
    # 打印 macro avg
    if 'macro avg' in metrics['classification_report']:
        ma = metrics['classification_report']['macro avg']
        print(f"{'-'*70}")
        print(f"{'macro avg':12} {ma['precision']:>12.4f} {ma['recall']:>12.4f} {ma['f1-score']:>12.4f} {int(ma['support']):>10}")
    print(f"{'='*70}")


def evaluate_test_set(model_path=None, fraud_type_model_path=None):
    """
    在测试集上评估模型
    """
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_df = load_and_preprocess_data(config.TEST_FILE)
    print(f"测试集样本数: {len(test_df)}")
    print(f"诈骗样本: {(test_df['is_fraud_binary'] == 1).sum()}")
    print(f"非诈骗样本: {(test_df['is_fraud_binary'] == 0).sum()}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG['model_name'])
    
    # 创建测试数据集
    test_dataset = FraudDataset(test_df.reset_index(drop=True), tokenizer, for_fraud_type=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.MODEL_CONFIG['batch_size'])
    
    # 仅诈骗样本的数据集
    fraud_df = test_df[test_df['is_fraud_binary'] == 1].reset_index(drop=True)
    fraud_dataset = FraudDataset(fraud_df, tokenizer, for_fraud_type=True)
    fraud_loader = torch.utils.data.DataLoader(fraud_dataset, batch_size=config.MODEL_CONFIG['batch_size'])
    
    # 创建输出目录
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    
    results = {}
    binary_metrics = None
    fraud_metrics = None
    difficulty_metrics = None
    
    # 评估二分类模型
    if model_path and os.path.exists(model_path):
        print(f"\n加载虚假通话二分类模型: {model_path}")
        model = MultiTaskModel(model_name=config.MODEL_CONFIG['model_name'], num_labels=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        binary_metrics, preds, labels = evaluate_binary_classification(model, test_loader, device)
        
        # 构建各类别的详细指标
        per_class_binary = {}
        for cls in ['非诈骗', '诈骗']:
            if cls in binary_metrics['classification_report']:
                r = binary_metrics['classification_report'][cls]
                per_class_binary[cls] = {
                    'precision': r['precision'],
                    'recall': r['recall'],
                    'f1-score': r['f1-score'],
                    'support': r['support'],
                }
        
        results['binary_classification'] = {
            'total': {
                'accuracy': binary_metrics['accuracy'],
                'precision': binary_metrics['precision_weighted'],
                'recall': binary_metrics['recall_weighted'],
                'f1-score': binary_metrics['f1_weighted'],
            },
            'per_class': per_class_binary,
            'confusion_matrix': binary_metrics['confusion_matrix']
        }
        
        print_binary_test_result(binary_metrics)
    
    # 评估诈骗类型多分类模型
    if fraud_type_model_path and os.path.exists(fraud_type_model_path):
        print(f"\n加载诈骗类型多分类模型: {fraud_type_model_path}")
        fraud_model = FraudDetectionModel(num_fraud_types=len(config.FRAUD_TYPES))
        fraud_model.load_state_dict(torch.load(fraud_type_model_path, map_location=device))
        fraud_model.to(device)
        
        if len(fraud_df) > 0:
            fraud_metrics, fraud_preds, fraud_labels = evaluate_fraud_type(fraud_model, fraud_loader, device)
            
            # 构建各类别的详细指标
            per_class_fraud = {}
            for fraud_type in config.FRAUD_TYPES:
                if fraud_type in fraud_metrics['classification_report']:
                    r = fraud_metrics['classification_report'][fraud_type]
                    per_class_fraud[fraud_type] = {
                        'precision': r['precision'],
                        'recall': r['recall'],
                        'f1-score': r['f1-score'],
                        'support': r['support'],
                    }
            
            # macro avg
            macro_avg = {}
            if 'macro avg' in fraud_metrics['classification_report']:
                ma = fraud_metrics['classification_report']['macro avg']
                macro_avg = {
                    'precision': ma['precision'],
                    'recall': ma['recall'],
                    'f1-score': ma['f1-score'],
                    'support': ma['support'],
                }
            
            # 按难度评估
            difficulty_metrics = evaluate_fraud_type_by_difficulty(
                fraud_model, fraud_df, tokenizer, device, 
                batch_size=config.MODEL_CONFIG['batch_size']
            )
            
            results['fraud_type_classification'] = {
                'total': {
                    'accuracy': fraud_metrics['accuracy'],
                    'precision': fraud_metrics['precision_weighted'],
                    'recall': fraud_metrics['recall_weighted'],
                    'f1-score': fraud_metrics['f1_weighted'],
                },
                'macro_avg': macro_avg,
                'per_class': per_class_fraud,
                'by_difficulty': difficulty_metrics,
            }
            
            print_fraud_type_test_result(fraud_metrics, difficulty_metrics)
    
    # 保存测试报告
    report_path = os.path.join(config.REPORT_DIR, 'test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存文本报告
    report_text_path = os.path.join(config.REPORT_DIR, 'test_report.txt')
    with open(report_text_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("诈骗电话检测模型 - 测试集评估报告\n")
        f.write("="*70 + "\n\n")
        
        if binary_metrics is not None:
            f.write("【二分类评估结果 (是否诈骗)】\n")
            f.write("\n总体结果:\n")
            f.write(f"Accuracy:  {binary_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {binary_metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall:    {binary_metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-score:  {binary_metrics['f1_weighted']:.4f}\n")
            f.write("\n具体类别结果:\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{'':10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}\n")
            f.write(f"{'-'*60}\n")
            for cls in ['非诈骗', '诈骗']:
                if cls in binary_metrics['classification_report']:
                    r = binary_metrics['classification_report'][cls]
                    f.write(f"{cls:10} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}\n")
            f.write(f"{'-'*60}\n\n")
        
        if fraud_metrics is not None:
            f.write("【多分类评估结果】\n")
            f.write("\n总体结果:\n")
            f.write(f"Accuracy:  {fraud_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {fraud_metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall:    {fraud_metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-score:  {fraud_metrics['f1_weighted']:.4f}\n")
            
            # 按难度输出
            if difficulty_metrics:
                difficulty_names = {'easy': '简单', 'medium': '中等', 'hard': '困难'}
                f.write("\n按难度分类结果:\n")
                f.write(f"{'-'*60}\n")
                f.write(f"{'难度':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}\n")
                f.write(f"{'-'*60}\n")
                for diff in ['easy', 'medium', 'hard']:
                    if diff in difficulty_metrics:
                        m = difficulty_metrics[diff]
                        f.write(f"{difficulty_names[diff]:>10} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['support']:>10}\n")
                f.write(f"{'-'*60}\n")
            
            f.write("\n具体类别结果:\n")
            f.write(f"{'-'*70}\n")
            f.write(f"{'':12} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Support':>10}\n")
            f.write(f"{'-'*70}\n")
            for fraud_type in config.FRAUD_TYPES:
                if fraud_type in fraud_metrics['classification_report']:
                    r = fraud_metrics['classification_report'][fraud_type]
                    f.write(f"{fraud_type:12} {r['precision']:>12.4f} {r['recall']:>12.4f} {r['f1-score']:>12.4f} {int(r['support']):>10}\n")
            if 'macro avg' in fraud_metrics['classification_report']:
                ma = fraud_metrics['classification_report']['macro avg']
                f.write(f"{'-'*70}\n")
                f.write(f"{'macro avg':12} {ma['precision']:>12.4f} {ma['recall']:>12.4f} {ma['f1-score']:>12.4f} {int(ma['support']):>10}\n")
            f.write(f"{'-'*70}\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"报告生成时间: {pd.Timestamp.now()}\n")
    
    print(f"\n测试报告已保存到: {report_path}")
    print(f"文本报告已保存到: {report_text_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估诈骗电话检测模型')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='二分类模型路径')
    parser.add_argument('--fraud_type_model_path', type=str, default=None,
                       help='诈骗类型模型路径')
    args = parser.parse_args()
    
    # 默认使用训练保存的模型
    if args.model_path is None:
        args.model_path = os.path.join(config.MODEL_SAVE_DIR, 'model_final.pt')
    if args.fraud_type_model_path is None:
        args.fraud_type_model_path = os.path.join(config.MODEL_SAVE_DIR, 'fraud_type_model.pt')
    
    results = evaluate_test_set(args.model_path, args.fraud_type_model_path)
