"""
evaluate_all.py — 批量评估 5 个测试集（模型只加载一次）
========================================================
基于 evaluator 已有函数，模型/分词器在开头加载一次，
循环切换数据路径即可。
"""

import os
import json
import torch
import pandas as pd
from transformers import BertTokenizer

import config
from evaluator import (set_seed, evaluate_binary_classification,
                       evaluate_fraud_type, evaluate_fraud_type_by_difficulty,
                       print_binary_test_result, print_fraud_type_test_result)
from model import FraudDetectionModel, MultiTaskModel
from dataset import load_and_preprocess_data, FraudDataset


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    data_dir = config.DATA_DIR
    output_base = os.path.join(config.PROJECT_ROOT, 'output-2')
    bs = config.MODEL_CONFIG['batch_size']

    # ---------- 模型 / 分词器只加载一次 ----------
    print("加载分词器……")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG['model_name'])

    print("加载模型……")
    bin_path = os.path.join(config.MODEL_SAVE_DIR, 'binary_model.pt')
    binary_model = MultiTaskModel(model_name=config.MODEL_CONFIG['model_name'], num_labels=2)
    binary_model.load_state_dict(torch.load(bin_path, map_location=device))
    binary_model.to(device); binary_model.eval()

    ft_path = os.path.join(config.MODEL_SAVE_DIR, 'fraud_type_model.pt')
    fraud_model = FraudDetectionModel(num_fraud_types=len(config.FRAUD_TYPES))
    fraud_model.load_state_dict(torch.load(ft_path, map_location=device))
    fraud_model.to(device); fraud_model.eval()

    # ---------- 逐个测试集 ----------
    datasets = [
        ('original',  'test.csv',          '原始测试集'),
        ('trust',     'test_trust.csv',    '信任策略增强'),
        ('urgency',   'test_urgency.csv',  '紧迫感策略增强'),
        ('emotional', 'test_emotional.csv','情感操纵策略增强'),
        ('combined',  'test_combined.csv', '三策略组合增强'),
    ]

    summary = {}

    for key, filename, desc in datasets:
        out_dir = os.path.join(output_base, key)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}\n[{key}] {desc}\n{'='*60}")

        # 加载数据
        test_df = load_and_preprocess_data(os.path.join(data_dir, filename))

        # DataLoader
        full_ds = FraudDataset(test_df.reset_index(drop=True), tokenizer, for_fraud_type=True)
        full_loader = torch.utils.data.DataLoader(full_ds, batch_size=bs)

        fraud_df = test_df[test_df['is_fraud_binary'] == 1].reset_index(drop=True)
        fraud_ds = FraudDataset(fraud_df, tokenizer, for_fraud_type=True)
        fraud_loader = torch.utils.data.DataLoader(fraud_ds, batch_size=bs)

        # 二分类
        bin_m, _, _ = evaluate_binary_classification(binary_model, full_loader, device)
        print_binary_test_result(bin_m)

        # 多分类
        fraud_m, diff_m = None, None
        if len(fraud_df) > 0:
            fraud_m, _, _ = evaluate_fraud_type(fraud_model, fraud_loader, device)
            diff_m = evaluate_fraud_type_by_difficulty(
                fraud_model, fraud_df, tokenizer, device, bs)
            print_fraud_type_test_result(fraud_m, diff_m)

        # 保存报告 (沿用 evaluator 报告格式)
        _save_report(bin_m, fraud_m, diff_m, test_df, out_dir, desc, config.FRAUD_TYPES)

        # 汇总
        e = {'description': desc, 'total_samples': len(test_df)}
        if bin_m:
            e['binary'] = {'accuracy': bin_m['accuracy'], 'precision': bin_m['precision_weighted'],
                           'recall': bin_m['recall_weighted'], 'f1': bin_m['f1_weighted']}
        if fraud_m:
            e['fraud_type'] = {'accuracy': fraud_m['accuracy'], 'precision': fraud_m['precision_weighted'],
                               'recall': fraud_m['recall_weighted'], 'f1': fraud_m['f1_weighted']}
        summary[key] = e

    # ---------- 终端汇总 ----------
    print(f"\n{'='*80}\n评估结果汇总\n{'='*80}")
    print(f"{'测试集':<14} {'二分类Acc':>10} {'二分类F1':>10} {'多分类Acc':>10} {'多分类F1':>10}\n{'-'*80}")
    for key, e in summary.items():
        b, f = e.get('binary', {}), e.get('fraud_type', {})
        print(f"{e['description']:<14} {b.get('accuracy',0):>10.4f} {b.get('f1',0):>10.4f} "
              f"{f.get('accuracy',0):>10.4f} {f.get('f1',0):>10.4f}")
    print(f"{'='*80}")

    with open(os.path.join(output_base, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {output_base}/summary.json")
    for key in summary:
        print(f"  {output_base}/{key}/  →  test_report.json  test_report.txt")


def _save_report(bin_m, fraud_m, diff_m, test_df, out_dir, desc, fraud_types):
    """保存 JSON 和 TXT 报告（与 evaluator 格式一致）"""
    # JSON
    results = {}
    if bin_m:
        pc = {}
        for cls in ['非诈骗', '诈骗']:
            if cls in bin_m['classification_report']:
                r = bin_m['classification_report'][cls]
                pc[cls] = {k: r[k] for k in ('precision', 'recall', 'f1-score', 'support')}
        results['binary_classification'] = {
            'total': {'accuracy': bin_m['accuracy'], 'precision': bin_m['precision_weighted'],
                      'recall': bin_m['recall_weighted'], 'f1-score': bin_m['f1_weighted']},
            'per_class': pc, 'confusion_matrix': bin_m['confusion_matrix']}
    if fraud_m:
        pc = {}
        for ft in fraud_types:
            if ft in fraud_m['classification_report']:
                r = fraud_m['classification_report'][ft]
                pc[ft] = {k: r[k] for k in ('precision', 'recall', 'f1-score', 'support')}
        ma = {}
        if 'macro avg' in fraud_m['classification_report']:
            r = fraud_m['classification_report']['macro avg']
            ma = {k: r[k] for k in ('precision', 'recall', 'f1-score', 'support')}
        results['fraud_type_classification'] = {
            'total': {'accuracy': fraud_m['accuracy'], 'precision': fraud_m['precision_weighted'],
                      'recall': fraud_m['recall_weighted'], 'f1-score': fraud_m['f1_weighted']},
            'macro_avg': ma, 'per_class': pc, 'by_difficulty': diff_m}
    with open(os.path.join(out_dir, 'test_report.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # TXT
    nf = int((test_df['is_fraud_binary'] == 1).sum())
    nn = int((test_df['is_fraud_binary'] == 0).sum())
    with open(os.path.join(out_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"诈骗电话检测模型 - 测试集评估报告\n测试集: {desc}\n")
        f.write(f"样本数: {len(test_df)} (诈骗:{nf}, 非诈骗:{nn})\n" + "=" * 70 + "\n\n")
        if bin_m:
            f.write("【二分类评估结果 (是否诈骗)】\n\n总体结果:\n")
            f.write(f"Accuracy:  {bin_m['accuracy']:.4f}\nPrecision: {bin_m['precision_weighted']:.4f}\n")
            f.write(f"Recall:    {bin_m['recall_weighted']:.4f}\nF1-score:  {bin_m['f1_weighted']:.4f}\n")
            f.write(f"\n具体类别结果:\n{'-'*60}\n{'':10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}\n{'-'*60}\n")
            for cls in ['非诈骗', '诈骗']:
                if cls in bin_m['classification_report']:
                    r = bin_m['classification_report'][cls]
                    f.write(f"{cls:10} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}\n")
            f.write(f"{'-'*60}\n\n")
        if fraud_m:
            f.write("【多分类评估结果 (诈骗类型)】\n\n总体结果:\n")
            f.write(f"Accuracy:  {fraud_m['accuracy']:.4f}\nPrecision: {fraud_m['precision_weighted']:.4f}\n")
            f.write(f"Recall:    {fraud_m['recall_weighted']:.4f}\nF1-score:  {fraud_m['f1_weighted']:.4f}\n")
            if diff_m:
                dnames = {'easy': '简单', 'medium': '中等', 'hard': '困难'}
                f.write(f"\n按难度:\n{'-'*60}\n{'难度':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Sup':>6}\n{'-'*60}\n")
                for d in ['easy', 'medium', 'hard']:
                    if d in diff_m:
                        m = diff_m[d]
                        f.write(f"{dnames[d]:>6} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['support']:>6}\n")
                f.write(f"{'-'*60}\n")
            f.write(f"\n具体类别结果:\n{'-'*70}\n{'':12} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Support':>10}\n{'-'*70}\n")
            for ft in fraud_types:
                if ft in fraud_m['classification_report']:
                    r = fraud_m['classification_report'][ft]
                    f.write(f"{ft:12} {r['precision']:>12.4f} {r['recall']:>12.4f} {r['f1-score']:>12.4f} {int(r['support']):>10}\n")
            if 'macro avg' in fraud_m['classification_report']:
                ma = fraud_m['classification_report']['macro avg']
                f.write(f"{'-'*70}\n{'macro avg':12} {ma['precision']:>12.4f} {ma['recall']:>12.4f} {ma['f1-score']:>12.4f} {int(ma['support']):>10}\n")
            f.write(f"{'-'*70}\n\n")
        f.write("=" * 70 + "\n" + f"报告生成时间: {pd.Timestamp.now()}\n")


if __name__ == '__main__':
    main()
