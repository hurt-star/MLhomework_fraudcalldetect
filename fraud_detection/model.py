"""
BERT模型定义
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import config


class FraudDetectionModel(nn.Module):
    """
    诈骗检测模型
    同时输出二分类(是否诈骗)和多分类(诈骗类型)
    """
    def __init__(self, model_name='bert-base-chinese', num_fraud_types=7, dropout=0.3):
        super(FraudDetectionModel, self).__init__()
        
        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 二分类分类器 (是否诈骗)
        self.is_fraud_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )
        
        # 多分类分类器 (诈骗类型)
        self.fraud_type_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_fraud_types)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取[CLS]位置的输出
        pooled_output = outputs.pooler_output
        
        # 二分类logits
        is_fraud_logits = self.is_fraud_classifier(pooled_output)
        
        # 多分类logits
        fraud_type_logits = self.fraud_type_classifier(pooled_output)
        
        return is_fraud_logits, fraud_type_logits


class MultiTaskModel(nn.Module):
    """
    多任务学习模型
    可以单独训练二分类或多分类任务
    """
    def __init__(self, model_name='bert-base-chinese', num_labels=2, dropout=0.3):
        super(MultiTaskModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.num_labels = num_labels
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def get_model(model_type='multitask', num_fraud_types=7):
    """
    获取模型实例
    """
    if model_type == 'multitask':
        model = MultiTaskModel(
            model_name=config.MODEL_CONFIG['model_name'],
            num_labels=2
        )
    elif model_type == 'fraud_detection':
        model = FraudDetectionModel(
            model_name=config.MODEL_CONFIG['model_name'],
            num_fraud_types=num_fraud_types
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试模型...")
    model = FraudDetectionModel(num_fraud_types=len(config.FRAUD_TYPES))
    
    # 创建假输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 21128, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # 前向传播
    is_fraud_logits, fraud_type_logits = model(input_ids, attention_mask)
    
    print(f"is_fraud_logits shape: {is_fraud_logits.shape}")  # [batch_size, 2]
    print(f"fraud_type_logits shape: {fraud_type_logits.shape}")  # [batch_size, num_fraud_types]
    
    print("模型测试成功!")
