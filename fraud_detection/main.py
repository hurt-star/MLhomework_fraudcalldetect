"""
主入口脚本
统一管理训练、评估、推理流程
"""
import argparse
import os
import sys


def main():
    """
    主入口
    """
    parser = argparse.ArgumentParser(
        description='诈骗电话检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  完整训练 (二分类 + 多分类):
    python main.py --mode train

  仅训练二分类:
    python main.py --mode train --task 1

  仅训练多分类:
    python main.py --mode train --task 2

  评估模型:
    python main.py --mode eval

  交互式预测:
    python main.py --mode predict --interactive

  演示预测:
    python main.py --mode predict --demo
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict', 'all'],
                       default='all', help='运行模式')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', '1', '2'],
                       help='训练任务: all(全部), 1(二分类), 2(多分类)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='二分类模型路径')
    parser.add_argument('--fraud_type_model_path', type=str, default=None,
                       help='诈骗类型模型路径')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式预测模式')
    parser.add_argument('--demo', action='store_true',
                       help='演示预测')
    
    args = parser.parse_args()
    
    # 训练模式
    if args.mode == 'train' or args.mode == 'all':
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        if args.mode == 'all':
            from trainer import train_all
            train_all()
        else:
            if args.task == 'all':
                from trainer import train_all
                train_all()
            elif args.task == '1':
                from trainer import train_binary_only
                train_binary_only()
            elif args.task == '2':
                from trainer import train_fraud_type_only
                train_fraud_type_only()
    
    # 评估模式
    if args.mode == 'eval' or args.mode == 'all':
        print("\n" + "="*60)
        print("开始评估")
        print("="*60)
        from evaluator import evaluate_test_set
        import config as cfg
        evaluate_test_set(
            args.model_path or os.path.join(cfg.MODEL_SAVE_DIR, 'binary_model.pt'),
            args.fraud_type_model_path or os.path.join(cfg.MODEL_SAVE_DIR, 'fraud_type_model.pt')
        )
    
    # 预测模式
    if args.mode == 'predict' or args.mode == 'all':
        from predictor import FraudDetector, interactive_predict, demo_predictions
        import config as cfg
        
        detector = FraudDetector(
            args.model_path or os.path.join(cfg.MODEL_SAVE_DIR, 'binary_model.pt'),
            args.fraud_type_model_path or os.path.join(cfg.MODEL_SAVE_DIR, 'fraud_type_model.pt')
        )
        
        if args.interactive:
            interactive_predict(args.model_path, args.fraud_type_model_path)
        elif args.demo or args.mode == 'all':
            demo_predictions(detector)
        else:
            demo_predictions(detector)
            print("\n如需交互式预测，请使用: python main.py --mode predict --interactive")


if __name__ == '__main__':
    main()
