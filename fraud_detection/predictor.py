"""
推理模块 - 单条对话预测
"""
import os
import re
import torch
from transformers import BertTokenizer

import config
from model import FraudDetectionModel, MultiTaskModel
from dataset import clean_dialogue


class FraudDetector:
    """
    诈骗电话检测器
    """
    def __init__(self, binary_model_path=None, fraud_type_model_path=None, device=None):
        """
        初始化检测器
        
        Args:
            binary_model_path: 二分类模型路径
            fraud_type_model_path: 诈骗类型模型路径
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG['model_name'])
        
        # 加载二分类模型
        self.binary_model = None
        if binary_model_path and os.path.exists(binary_model_path):
            print(f"加载二分类模型: {binary_model_path}")
            self.binary_model = MultiTaskModel(model_name=config.MODEL_CONFIG['model_name'])
            self.binary_model.load_state_dict(torch.load(binary_model_path, map_location=self.device))
            self.binary_model.to(self.device)
            self.binary_model.eval()
        
        # 加载诈骗类型模型
        self.fraud_type_model = None
        if fraud_type_model_path and os.path.exists(fraud_type_model_path):
            print(f"加载诈骗类型模型: {fraud_type_model_path}")
            self.fraud_type_model = FraudDetectionModel(num_fraud_types=len(config.FRAUD_TYPES))
            self.fraud_type_model.load_state_dict(torch.load(fraud_type_model_path, map_location=self.device))
            self.fraud_type_model.to(self.device)
            self.fraud_type_model.eval()
    
    def preprocess_text(self, text):
        """
        预处理输入文本
        """
        # 清洗文本
        cleaned_text = clean_dialogue(text)
        
        # 统一角色格式
        cleaned_text = cleaned_text.replace('left:', '客服:').replace('right:', '用户:')
        
        return cleaned_text
    
    def predict(self, dialogue_text):
        """
        预测单条对话
        
        Args:
            dialogue_text: 对话文本 (原始格式)
            
        Returns:
            dict: 预测结果
                - is_fraud: "TRUE" 或 "FALSE"
                - fraud_type: 诈骗类型 (如is_fraud为FALSE则为空)
                - confidence: 置信度
        """
        # 预处理
        cleaned_text = self.preprocess_dialogue(dialogue_text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            max_length=config.MODEL_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        result = {
            'is_fraud': 'FALSE',
            'fraud_type': '',
            'confidence': 0.0
        }
        
        with torch.no_grad():
            # 二分类预测
            if self.binary_model is not None:
                logits = self.binary_model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                fraud_prob = probs[0][1].item()  # 诈骗概率
                result['confidence'] = fraud_prob
                
                # 判断是否诈骗 (阈值0.5)
                if fraud_prob >= 0.5:
                    result['is_fraud'] = 'TRUE'
                else:
                    result['is_fraud'] = 'FALSE'
            
            # 诈骗类型预测
            if self.fraud_type_model is not None and result['is_fraud'] == 'TRUE':
                _, fraud_type_logits = self.fraud_type_model(input_ids, attention_mask)
                fraud_type_probs = torch.softmax(fraud_type_logits, dim=1)
                fraud_type_idx = torch.argmax(fraud_type_probs, dim=1).item()
                fraud_type_confidence = fraud_type_probs[0][fraud_type_idx].item()
                
                # 获取诈骗类型名称
                if 0 <= fraud_type_idx < len(config.FRAUD_TYPES):
                    result['fraud_type'] = config.FRAUD_TYPES[fraud_type_idx]
                    result['confidence'] = fraud_type_confidence
        
        return result
    
    def preprocess_dialogue(self, text):
        """
        预处理对话文本 (保留说话人信息)
        """
        # 移除音频内容标记
        text = re.sub(r'音频内容：', '', text)
        text = re.sub(r'\*\*', '', text)
        
        # 脱敏
        text = re.sub(r'\d{11}', '【手机号】', text)
        text = re.sub(r'\d{16,19}', '【银行卡号】', text)
        text = re.sub(r'\d{15}|\d{17}[\dXx]', '【身份证号】', text)
        text = re.sub(r'https?://[^\s]+', '【链接】', text)
        text = re.sub(r'验证码[是为：:\s]*[a-zA-Z0-9]{4,8}', '验证码【已隐藏】', text)
        
        # 统一角色格式
        # text = text.replace('left:', '客服:').replace('right:', '用户:')
        
        # 清理空白
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def predict_batch(self, dialogue_texts):
        """
        批量预测
        
        Args:
            dialogue_texts: 对话文本列表
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for text in dialogue_texts:
            results.append(self.predict(text))
        return results


def interactive_predict(binary_model_path=None, fraud_type_model_path=None):
    """
    交互式预测
    """
    detector = FraudDetector(binary_model_path, fraud_type_model_path)
    
    print("\n" + "="*50)
    print("诈骗电话检测 - 交互式预测")
    print("="*50)
    print("输入对话内容进行预测 (输入 'quit' 退出)")
    print("-"*50)
    
    while True:
        print("\n请输入对话内容 (可直接粘贴对话文本):")
        print("-"*50)
        
        lines = []
        while True:
            line = input()
            if line.strip() == '':
                break
            if line.lower() == 'quit':
                return
            lines.append(line)
        
        if not lines:
            continue
            
        dialogue_text = '\n'.join(lines)
        
        # 预测
        result = detector.predict(dialogue_text)
        
        # 输出结果
        print("\n" + "="*50)
        print("预测结果:")
        print(f'  "is_fraud": "{result["is_fraud"]}",')
        fraud_type = result["fraud_type"] if result["fraud_type"] else '""'
        print(f'  "fraud_type": "{fraud_type}",')
        print(f'  "confidence": {result["confidence"]:.4f}')
        print("="*50)


def demo_predictions(detector):
    """
    演示预测
    """
    demo_samples = [
        # 诈骗样本示例
        """left: 您好，我是XX彩票中心的工作人员，恭喜您被系统随机选中成为本期大奖的幸运得主！请您提供一下中奖彩票的后四位号码，以便我们核实您的身份并安排奖金发放。
right: 哎呀，真的吗？我怎么这么幸运呢。不过，我最近都没买彩票呀，是不是弄错了？
left: 请您仔细回忆一下，可能是您的朋友或者家人帮您购买的。我们系统显示您的号码确实是中奖号码，如果不及时兑奖可能会被取消资格。您可以先提供彩票号码，我们帮您查询具体情况。
right: 哦，这可真奇怪，我肯定没买过。不过，万一真是这样，那可真是太好了。你们能告诉我中了多少钱吗？我再想想是不是有这么回事。
left: 您中的是我们特别设置的"幸运大奖"，奖金高达五百万元！但根据规定，我们需要先核实您的身份信息和彩票号码，请您提供一下您的姓名和身份证号码，方便我们为您办理兑奖手续。
right: 五百万元？天哪，这太不可思议了。不过，我还是有点不太放心。你们能不能先给我一个官方的证明或者文件，这样我也可以确认一下。
left: 由于奖金数额较大，我们需要尽快核实您的信息，防止奖金被冒领。官方文件需要在您确认身份后才能发放。请您提供一下姓名和身份证号码，我们会为您加急处理。时间紧迫，耽误太久可能会有风险。
right: 我理解你们的担心，但我也需要确保这是真的。要不这样，我先给我儿子打个电话，让他帮我查一查。毕竟这么大一笔钱，总要谨慎一些。
left: 好的，我理解您的顾虑，但请您尽快处理，因为中奖信息保留时间有限。如果您确认无误，我们会立即为您办理兑奖手续。希望您能抓紧时间，免得错过这个机会。如果耽误太久，我们就无法保证中奖资格了。""",
        
        # 非诈骗样本示例
        """left: 喂，你好，我是美食广场的客服，你之前在我们这里订过餐，不知道你对我们餐厅的服务和菜品满意吗？
right: 哦，你好，挺好的，服务不错，菜品也挺好吃的。
left: 太好了，听到你满意的反馈我们也很高兴。不知道你有没有兴趣参加我们即将举办的一次美食节活动？
right: 美食节？听起来不错，能告诉我更多关于这个活动的信息吗？
left: 当然可以。这次美食节，我们将邀请来自不同国家的厨师，提供各式各样的美食。还有特别优惠，比如买一送一，部分菜品打折。
right: 哇，听起来真的很不错，活动是什么时候开始呢？
left: 美食节将在下周六上午十点正式开始，持续到晚上九点。你有时间来参加吗？
right: 应该可以，不过我还想知道，如果带朋友一起去，有没有什么优惠呢？
left: 带上你的朋友来参加美食节，可以享受额外的二十元代金券，用于任何消费。
right: 太好了，那我一定会带朋友一起去的。谢谢你的信息。
left: 不客气，我们美食广场期待你的光临。如果还有其他问题，随时欢迎咨询。""",

        """left: 喂，你好，这里是深圳信用贷款中心，我是经理李明，根据你公司的资质，我们为你提供一笔无抵押贷款，有兴趣了解一下吗？
right: 好啊，我最近确实有点资金周转的问题。
left: 太好了，你只需要提供一些基本的资料，我们就能帮你快速审批。不过为了安全起见，我们需要你先下载我们公司的专用APP，这样可以更加便捷地操作。
right: 下载APP？这样会不会不太安全？
left: 不用担心，我们的APP是经过严格安全测试的，你可以放心使用。下载后，你还可以享受更多优惠。
right: 那好吧，我试试看。
left: 太好了，你直接打开这个链接就可以下载了。下载后记得注册并绑定你的银行卡，这样我们才能更快地为你放款。
right: 好的，我知道了。谢谢。
left: 不客气，如果有任何问题，随时联系我。祝你一切顺利，再见。
right: 再见。"""
    ]
    
    print("\n" + "="*60)
    print("演示预测结果")
    print("="*60)
    
    for i, sample in enumerate(demo_samples):
        print(f"\n【示例 {i+1}】")
        print("-"*40)
        # 只显示前100个字符
        display_text = sample[:100] + "..." if len(sample) > 100 else sample
        print(display_text)
        print("-"*40)
        
        result = detector.predict(sample)
        
        print(f'  "is_fraud": "{result["is_fraud"]}",')
        fraud_type = result["fraud_type"] if result["fraud_type"] else '""'
        print(f'  "fraud_type": "{fraud_type}",')
        print(f'  "confidence": {result["confidence"]:.4f}')
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='诈骗电话检测推理')
    parser.add_argument('--model_path', type=str, default=None,
                       help='二分类模型路径')
    parser.add_argument('--fraud_type_model_path', type=str, default=None,
                       help='诈骗类型模型路径')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式预测模式')
    parser.add_argument('--demo', action='store_true',
                       help='演示预测')
    args = parser.parse_args()
    
    # 默认模型路径
    if args.model_path is None:
        args.model_path = os.path.join(config.MODEL_SAVE_DIR, 'model_final.pt')
    if args.fraud_type_model_path is None:
        args.fraud_type_model_path = os.path.join(config.MODEL_SAVE_DIR, 'fraud_type_model.pt')
    
    detector = FraudDetector(args.model_path, args.fraud_type_model_path)
    
    if args.interactive:
        interactive_predict(args.model_path, args.fraud_type_model_path)
    elif args.demo:
        demo_predictions(detector)
    else:
        # 默认执行演示
        demo_predictions(detector)
        
        print("\n如需交互式预测，请使用: python predictor.py --interactive")
        print("如需评估模型，请使用: python evaluator.py")
        print("如需训练模型，请使用: python trainer.py")
