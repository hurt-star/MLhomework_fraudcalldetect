# 《机器学习初步》课程作业：基于Fraud-R1的虚假通话检测

基于 BERT 中文预训练模型的电话诈骗检测系统，以 [Fraud-R1 (ACL 2025)](https://aclanthology.org/2025.findings-acl.226/) 为数据构建增强方法，研究"建立信任""制造紧迫感""情感操纵"三类社会工程策略对模型检测能力的干扰，并通过课程学习（Curriculum Learning）提升鲁棒性。

使用 AI 生成项目主要代码，原始数据集为课程提供。

原项目在 Cloud Studio 内完成，链接： https://cloudstudio.net/a/35108604608548864?channel=share&sharetype=URL

## 核心特点

- **双任务分类**：二分类（是否诈骗）+ 多分类（诈骗类型识别）
- **课程学习**：按对话轮数将样本分为 Easy / Medium / Hard 三阶段渐进训练
- **增强测试**：基于 Fraud-R1 的社会工程策略（信任/紧迫/情感/组合）构造增强测试集

## 项目结构

```
.
├── data/                              # 数据集
│   ├── train.csv                      # 训练集
│   ├── test.csv                       # 原始测试集
│   ├── test_trust.csv                 # 信任策略增强测试集
│   ├── test_urgency.csv               # 紧迫感策略增强测试集
│   ├── test_emotional.csv             # 情感操纵策略增强测试集
│   ├── test_combined.csv              # 三策略组合增强测试集
│   └── 说明.txt                       # 数据字段说明
│
├── fraud_detection/                   # 核心代码
│   ├── config.py                      # 全局配置（模型/路径/超参/标签）
│   ├── model.py                       # BERT 双头模型定义
│   ├── dataset.py                     # 数据加载、预处理、脱敏、课程学习划分
│   ├── trainer.py                     # 三阶段课程学习训练逻辑
│   ├── evaluator.py                   # 二分类/多分类评估 + 按难度分层报告
│   ├── predictor.py                   # 单条/批量推理
│   ├── main.py                        # 统一入口（训练/评估/推理）
│   ├── augment_trust.py               # 策略增强：建立信任
│   ├── augment_urgency.py             # 策略增强：制造紧迫感
│   ├── augment_emotional.py           # 策略增强：情感操纵
│   ├── augment_combined.py            # 策略增强：三策略叠加 + Excel 对比
│   ├── evaluate_all.py                # 批量评估 5 个测试集
│   ├── count_diff.py                  # 测试集数据统计
│   ├── requirements.txt               # Python 依赖
│   ├── output/reports/                # 实验1结果，训练/测试评估报告
│   └── output-2/                      # 实验2结果，各测试集评估报告
│       ├── original/
│       ├── trust/
│       ├── urgency/
│       ├── emotional/
│       ├── combined/
│       └── summary.json
│
└── .gitignore
```

## 数据格式

CSV 文件包含以下字段：

| 字段 | 说明 |
|------|------|
| `specific_dialogue_content` | 对话文本，格式为 `left:` `right:` 交替的多轮对话 |
| `interaction_strategy` | 五种对话策略标签（事实真实性/完整性/清晰度/相关性/个性化） |
| `call_type` | 通话类型（咨询客服、预约服务等） |
| `is_fraud` | 是否欺诈（True/False） |
| `fraud_type` | 欺诈类型（客服诈骗/银行诈骗/钓鱼诈骗/投资诈骗/绑架诈骗/身份盗窃/彩票诈骗） |

## 方法

### 课程学习策略

按对话轮数划分难度，分三阶段渐进训练：

| 阶段 | 样本范围 | 轮数阈值 |
|------|---------|----------|
| Stage 1 (Easy) | 简单样本 | ≤ 7 轮 |
| Stage 2 (Medium) | 简单 + 中等 | ≤ 15 轮 |
| Stage 3 (Hard) | 全部样本 | 全部 |

### 数据增强策略

对原始测试集中的诈骗样本，应用以下社会工程策略进行改写增强：

| 策略 | 核心手段 | 示例 |
|------|---------|------|
| **建立信任** | 添加工号、执业编号、可验证渠道、制度背书 | "我的工号是 8823，您可以拨打 95599 核实" |
| **制造紧迫感** | 时效限制、稀缺性施压、最终期限提醒 | "该优惠仅剩 2 小时，逾期将恢复原价" |
| **情感操纵** | 贪婪引诱、恐惧施压、同情求助、关心绑架 | "您的账户存在异常交易，如不及时处理将造成资金损失" |
| **组合增强** | 依次叠加信任 → 紧迫 → 情感 | 同时包含以上三类话术 |

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd <project-dir>

# 安装依赖
pip install -r fraud_detection/requirements.txt
```

**依赖项**：Python 3.8+ / PyTorch ≥ 1.9 / transformers ≥ 4.36 / pandas / scikit-learn / tqdm

首次运行时会自动下载 `bert-base-chinese` 预训练模型。

## 使用方式

### 1. 训练模型

```bash
cd fraud_detection

# 完整训练（二分类 + 多分类）
python main.py --mode train

# 仅训练二分类
python main.py --mode train --task 1

# 仅训练多分类（诈骗类型）
python main.py --mode train --task 2
```

### 2. 评估模型

```bash
# 在原始测试集上评估
python main.py --mode eval

# 指定模型路径
python main.py --mode eval \
  --model_path output/models/binary_model.pt \
  --fraud_type_model_path output/models/fraud_type_model.pt
```

### 3. 构造增强测试集

```bash
# 生成信任策略增强测试集
python augment_trust.py

# 生成紧迫感策略增强测试集
python augment_urgency.py

# 生成情感操纵策略增强测试集
python augment_emotional.py

# 生成三策略组合增强测试集 + Excel 对比文件
python augment_combined.py
```

### 4. 批量评估所有测试集

```bash
# 在 5 个测试集（原始 + 4 种增强）上批量评估，模型仅加载一次
python evaluate_all.py
```

### 5. 推理预测

```bash
# 演示预测（内置样本）
python main.py --mode predict --demo

# 交互式预测（已弃用，暂不可用）
python main.py --mode predict --interactive
```

## 实验结果

### 二分类（是否诈骗） — 各测试集表现

| 测试集 | Accuracy | F1-score |
|--------|----------|----------|
| 原始测试集 | 1.0000 | 1.0000 |
| 信任策略增强 | 1.0000 | 1.0000 |
| 紧迫感策略增强 | 0.9996 | 0.9996 |
| 情感操纵策略增强 | 1.0000 | 1.0000 |
| 三策略组合增强 | 1.0000 | 1.0000 |

### 多分类（诈骗类型识别） — 各测试集表现

| 测试集 | Accuracy | F1-score |
|--------|----------|----------|
| 原始测试集 | 0.7678 | 0.7645 |
| 信任策略增强 | 0.6994 | 0.6704 |
| 紧迫感策略增强 | 0.7628 | 0.7542 |
| 情感操纵策略增强 | 0.7534 | 0.7475 |
| 三策略组合增强 | 0.6936 | 0.6643 |

> **结论**：BERT 模型对是否诈骗识别效果极佳，对诈骗类型识别在增强测试集上性能出现下降，但影响轻微，可能因数据增强质量一般，其为通过代码批量替换，可能影响实验效果。

> **局限**：未使用大模型训练测试，使用的是传统文本模型（~~实为时间与财力不足~~）；数据增强简单使用代码替换，质量一般，应使用大模型基于一定策略逐一改写；严格来说未使用近年较新颖的训练/检测方法（~~前期研读论文研错方向了~~），只能将就一下就这样吧。

## 参考文献

1. *Fraud-R1: A Multi-Round Phone Scam Dataset with Social Engineering Manipulations.* Findings of ACL 2025. [论文链接](https://aclanthology.org/2025.findings-acl.226/)

## 声明

本项目为个人课程作业，仅用于日常学习交流，严禁用于抄袭或其他违法活动。
