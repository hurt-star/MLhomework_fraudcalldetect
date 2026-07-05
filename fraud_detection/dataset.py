"""
数据预处理模块
"""
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import config


def clean_dialogue(text):
    """
    清洗对话文本
    - 去除噪声
    - 脱敏手机号、银行卡、链接等敏感信息
    """
    if pd.isna(text):
        return None
    
    # 移除音频内容标记
    text = re.sub(r'音频内容：', '', text)
    text = re.sub(r'\*\*', '', text)
    
    # 脱敏手机号 (11位数字)
    text = re.sub(r'\d{11}', '【手机号】', text)
    # 脱敏银行卡号 (16-19位数字)
    text = re.sub(r'\d{16,19}', '【银行卡号】', text)
    # 脱敏身份证号 (15或18位)
    text = re.sub(r'\d{15}|\d{17}[\dXx]', '【身份证号】', text)
    # 脱敏链接
    text = re.sub(r'https?://[^\s]+', '【链接】', text)
    text = re.sub(r'HTPS?[Uu][Dd][Aa][Ii][Bb][Oo][Aa][Oo][Cc][Oo][Mm][^\s]*', '【链接】', text)
    # 脱敏验证码
    text = re.sub(r'验证码[是为：:\s]*[a-zA-Z0-9]{4,8}', '验证码【已隐藏】', text)
    
    # 统一角色格式: left → 客服, right → 用户
    # text = text.replace('left:', '客服:').replace('right:', '用户:')
    
    # 清理多余空白
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def count_dialogue_turns(text):
    """
    统计对话轮数
    通过统计"客服:"和"用户:"的出现次数
    """
    if pd.isna(text):
        return 0
    # 统计客服和用户的发言次数
    left_count = len(re.findall(r'left:', text.lower()))
    right_count = len(re.findall(r'right:', text.lower()))
    return left_count + right_count


def assign_difficulty(turn_count):
    """
    根据对话轮数划分课程学习难度
    easy: 6轮及以内
    medium: 7-14轮
    hard: 15轮及以上
    """
    if turn_count <= config.CURRICULUM_RULES['easy_threshold']:
        return 'easy'
    elif turn_count <= config.CURRICULUM_RULES['medium_threshold']:
        return 'medium'
    else:
        return 'hard'


def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    df = pd.read_csv(file_path)
    
    # 重命名列
    df.columns = df.columns.str.strip()
    
    # 清洗对话文本
    df['cleaned_text'] = df['specific_dialogue_content'].apply(clean_dialogue)
    
    # 统计对话轮数
    df['turn_count'] = df['specific_dialogue_content'].apply(count_dialogue_turns)
    
    # 划分难度
    df['difficulty'] = df['turn_count'].apply(assign_difficulty)
    
    # 处理is_fraud标签
    df['is_fraud'] = df['is_fraud'].astype(str).str.upper().str.strip()
    df['is_fraud_binary'] = (df['is_fraud'] == 'TRUE').astype(int)
    
    # 处理fraud_type标签 (仅当is_fraud为True时有效)
    df['fraud_type'] = df['fraud_type'].fillna('').astype(str).str.strip()
    # 将空白和"nan"映射为空字符串
    df.loc[df['fraud_type'] == 'nan', 'fraud_type'] = ''
    
    # 创建多分类标签
    fraud_type_to_idx = {ft: idx for idx, ft in enumerate(config.FRAUD_TYPES)}
    # 对于非诈骗样本，fraud_type为空，多分类标签设为-1
    df['fraud_type_idx'] = df.apply(
        lambda x: fraud_type_to_idx.get(x['fraud_type'], -1) if x['is_fraud'] == 'TRUE' else -1,
        axis=1
    )
    
    # 过滤无效样本 (缺少必要标签)
    valid_df = df.dropna(subset=['cleaned_text', 'is_fraud'])
    valid_df = valid_df[valid_df['is_fraud'].isin(['TRUE', 'FALSE'])]
    
    print(f"原始数据量: {len(df)}, 有效数据量: {len(valid_df)}")
    print(f"诈骗样本: {(valid_df['is_fraud_binary'] == 1).sum()}")
    print(f"非诈骗样本: {(valid_df['is_fraud_binary'] == 0).sum()}")
    print(f"难度分布: easy={len(valid_df[valid_df['difficulty']=='easy'])}, "
          f"medium={len(valid_df[valid_df['difficulty']=='medium'])}, "
          f"hard={len(valid_df[valid_df['difficulty']=='hard'])}")
    
    return valid_df


class FraudDataset(Dataset):
    """
    诈骗检测数据集
    """
    def __init__(self, dataframe, tokenizer, max_length=512, for_fraud_type=False):
        self.texts = dataframe['cleaned_text'].tolist()
        self.is_fraud_labels = dataframe['is_fraud_binary'].tolist()
        self.fraud_type_labels = dataframe['fraud_type_idx'].tolist()
        self.for_fraud_type = for_fraud_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'is_fraud_label': torch.tensor(self.is_fraud_labels[idx], dtype=torch.long),
        }
        
        # 多分类标签
        if self.for_fraud_type:
            item['fraud_type_label'] = torch.tensor(self.fraud_type_labels[idx], dtype=torch.long)
        
        return item


def prepare_dataloaders(df, tokenizer, batch_size=16, eval_ratio=0.05):
    """
    准备训练和评估数据加载器
    """
    # 划分训练集和评估集
    train_df, eval_df = train_test_split(df, test_size=eval_ratio, random_state=42, stratify=df['is_fraud_binary'])
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    
    # 创建数据集 (用于二分类)
    train_dataset = FraudDataset(train_df, tokenizer)
    eval_dataset = FraudDataset(eval_df, tokenizer)
    
    # 创建数据集 (用于多分类 - 仅诈骗样本)
    train_fraud_df = train_df[train_df['is_fraud_binary'] == 1]
    eval_fraud_df = eval_df[eval_df['is_fraud_binary'] == 1]
    
    train_fraud_dataset = FraudDataset(train_fraud_df.reset_index(drop=True), tokenizer, for_fraud_type=True)
    eval_fraud_dataset = FraudDataset(eval_fraud_df.reset_index(drop=True), tokenizer, for_fraud_type=True)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
    train_fraud_loader = torch.utils.data.DataLoader(train_fraud_dataset, batch_size=batch_size, shuffle=True)
    eval_fraud_loader = torch.utils.data.DataLoader(eval_fraud_dataset, batch_size=batch_size)
    
    return {
        'train_df': train_df,
        'eval_df': eval_df,
        'train_loader': train_loader,
        'eval_loader': eval_loader,
        'train_fraud_loader': train_fraud_loader,
        'eval_fraud_loader': eval_fraud_loader,
    }


def prepare_curriculum_dataloaders(df, tokenizer, batch_size=16, eval_ratio=0.05, for_fraud_type=False):
    """
    准备课程学习数据加载器
    按难度划分数据集
    
    Args:
        df: 数据DataFrame
        tokenizer: 分词器
        batch_size: 批大小
        eval_ratio: 验证集比例
        for_fraud_type: 是否为诈骗类型多分类任务
    """
    # 多分类任务：只使用诈骗样本（非诈骗样本的fraud_type_idx=-1，无效）
    if for_fraud_type:
        df = df[df['fraud_type_idx'] >= 0].reset_index(drop=True)
        stratify_col = 'fraud_type_idx'
        print(f"\n多分类任务 - 诈骗样本数量: {len(df)}")
    else:
        stratify_col = 'is_fraud_binary'
    
    easy_df = df[df['difficulty'] == 'easy'].reset_index(drop=True)
    medium_df = df[df['difficulty'] == 'medium'].reset_index(drop=True)
    hard_df = df[df['difficulty'] == 'hard'].reset_index(drop=True)
    
    print(f"难度分布: easy={len(easy_df)}, medium={len(medium_df)}, hard={len(hard_df)}")
    
    def safe_split(data_df, strat_col, eval_ratio):
        """安全的数据划分，处理样本太少无法分层的情况"""
        try:
            return train_test_split(data_df, test_size=eval_ratio, random_state=42, stratify=data_df[strat_col])
        except ValueError:
            # 分层失败时使用普通划分
            return train_test_split(data_df, test_size=eval_ratio, random_state=42)
    
    # 每个难度级别都划分训练集和评估集
    train_easy, eval_easy = safe_split(easy_df, stratify_col, eval_ratio)
    train_medium, eval_medium = safe_split(medium_df, stratify_col, eval_ratio)
    train_hard, eval_hard = safe_split(hard_df, stratify_col, eval_ratio)
    
    # 创建数据集
    datasets = {}
    for name, subset in [('easy', (train_easy, eval_easy)), 
                         ('medium', (train_medium, eval_medium)), 
                         ('hard', (train_hard, eval_hard))]:
        train, eval_s = subset
        datasets[name] = {
            'train_loader': torch.utils.data.DataLoader(
                FraudDataset(train.reset_index(drop=True), tokenizer, for_fraud_type=for_fraud_type), 
                batch_size=batch_size, shuffle=True
            ),
            'eval_loader': torch.utils.data.DataLoader(
                FraudDataset(eval_s.reset_index(drop=True), tokenizer, for_fraud_type=for_fraud_type), 
                batch_size=batch_size
            ),
            'train_size': len(train),
            'eval_size': len(eval_s),
        }
    
    # 合并数据集用于后续阶段
    # Stage 2: easy + medium
    stage2_train = pd.concat([train_easy, train_medium]).reset_index(drop=True)
    stage2_eval = pd.concat([eval_easy, eval_medium]).reset_index(drop=True)
    datasets['stage2'] = {
        'train_loader': torch.utils.data.DataLoader(
            FraudDataset(stage2_train, tokenizer, for_fraud_type=for_fraud_type), 
            batch_size=batch_size, shuffle=True
        ),
        'eval_loader': torch.utils.data.DataLoader(
            FraudDataset(stage2_eval, tokenizer, for_fraud_type=for_fraud_type), 
            batch_size=batch_size
        ),
        'train_size': len(stage2_train),
        'eval_size': len(stage2_eval),
    }
    
    # Stage 3: 全部数据
    _, all_eval = safe_split(df, stratify_col, eval_ratio)
    datasets['stage3'] = {
        'train_loader': torch.utils.data.DataLoader(
            FraudDataset(df.reset_index(drop=True), tokenizer, for_fraud_type=for_fraud_type), 
            batch_size=batch_size, shuffle=True
        ),
        'eval_loader': torch.utils.data.DataLoader(
            FraudDataset(all_eval.reset_index(drop=True), tokenizer, for_fraud_type=for_fraud_type), 
            batch_size=batch_size
        ),
        'train_size': len(df),
        'eval_size': len(all_eval),
        'eval_df': all_eval.reset_index(drop=True),
    }
    
    return datasets


if __name__ == '__main__':
    # 测试数据预处理
    from transformers import BertTokenizer
    
    print("加载数据...")
    df = load_and_preprocess_data(config.TRAIN_FILE)
    
    print("\n加载分词器...")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG['model_name'])
    
    print("\n准备数据加载器...")
    dataloaders = prepare_curriculum_dataloaders(
        df, tokenizer, 
        batch_size=config.MODEL_CONFIG['batch_size'],
        eval_ratio=config.MODEL_CONFIG['eval_ratio']
    )
    
    print("\n各阶段数据统计:")
    for stage in ['easy', 'medium', 'hard', 'stage2', 'stage3']:
        if stage in dataloaders:
            print(f"  {stage}: 训练集 {dataloaders[stage]['train_size']}, 评估集 {dataloaders[stage]['eval_size']}")
