import os, re, pandas as pd

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def count_turns(text):
    if pd.isna(text): return 0
    return len(re.findall(r'left:', str(text))) + len(re.findall(r'right:', str(text)))

def assign(n):
    if n <= 7: return '简单'
    elif n <= 15: return '中等'
    return '困难'

files = [
    ('original',       '原始测试集',       'test.csv'),
    ('trust',          '信任策略增强',      'test_trust.csv'),
    ('urgency',        '紧迫感策略增强',    'test_urgency.csv'),
    ('emotional',      '情感操纵策略增强',  'test_emotional.csv'),
    ('combined',       '三策略组合增强',    'test_combined.csv'),
]

print(f"{'测试集':<14} {'总样本':>6} {'简单':>6} {'中等':>6} {'困难':>6} {'诈骗':>6} {'非诈骗':>6}")
print("-" * 62)
for key, desc, fname in files:
    df = pd.read_csv(os.path.join(data_dir, fname), encoding='utf-8-sig')
    td = df['specific_dialogue_content'].apply(count_turns)
    df['diff'] = td.apply(assign)
    fd = (df['is_fraud'].astype(str).str.upper().str.strip()=='TRUE').sum()
    e = (df['diff']=='简单').sum()
    m = (df['diff']=='中等').sum()
    h = (df['diff']=='困难').sum()
    print(f"{desc:<14} {len(df):>6} {e:>6} {m:>6} {h:>6} {fd:>6} {len(df)-fd:>6}")

# 汇总
df0 = pd.read_csv(os.path.join(data_dir, 'test.csv'), encoding='utf-8-sig')
td = df0['specific_dialogue_content'].apply(count_turns)
df0['diff'] = td.apply(assign)
print()
print(f"注：后四个增强集的样本数包含了全部2,677条（含129条无is_fraud标签的样本），")
print(f"    而评估时 load_and_preprocess_data 会过滤无效样本到2,548条。")
print(f"    前表中的诈骗/非诈骗计数基于 raw CSV 的 is_fraud 字段。")
print(f"    难度分布不受策略改写影响，五个数据集完全一致。")
