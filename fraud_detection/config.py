"""
项目配置文件
"""
import os

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# 模型配置
MODEL_CONFIG = {
    'model_name': 'bert-base-chinese',  # 中文BERT模型
    'max_length': 512,                  # 最大序列长度
    'batch_size': 16,                    # 批大小
    'learning_rate': 2e-5,              # 学习率
    'warmup_ratio': 0.1,                # 预热比例
    'weight_decay': 0.01,               # 权重衰减
    'eval_ratio': 0.1,                  # 评估集比例 (10%)
}

# 二分类训练配置 (课程学习)
BINARY_CONFIG = {
    'num_epochs_stage1': 1,  # Easy
    'num_epochs_stage2': 1,  # Easy + Medium
    'num_epochs_stage3': 1,  # All
}

# 多分类训练配置 (课程学习)
FRAUD_TYPE_CONFIG = {
    'num_epochs_stage1': 3,  # Easy
    'num_epochs_stage2': 3,  # Easy + Medium
    'num_epochs_stage3': 3,  # All
}

# 课程学习难度划分
CURRICULUM_RULES = {
    'easy_threshold': 7,      # 7轮及以内为easy
    'medium_threshold': 15,   # 8-15轮为medium
    # 16轮及以上为hard
}

# 诈骗类型标签
FRAUD_TYPES = [
    '客服诈骗',
    '银行诈骗',
    '钓鱼诈骗',
    '投资诈骗',
    '绑架诈骗',
    '身份盗窃',
    '彩票诈骗',
]

# 输出目录 (相对于项目根目录)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
