"""
augment_urgency.py — 策略2：制造紧迫感（Urgency Creation）
===========================================================
基于 Fraud-R1 诱导策略框架，对 test.csv 中每条对话进行内容感知的
紧迫感增强改写。根据对话语境和进展阶段，在合适位置自然嵌入
时间压力、名额限制、即时行动等元素。

改写原则：
  1. 在诈骗"解决方案"提出后立即附加时效限制
  2. 在目标对象犹豫时追加稀缺性施压
  3. 在对话收尾前追加最终期限提醒
  4. 紧迫语言必须与对话领域匹配（银行→审批截止 / 快递→理赔时限等）
  5. 非诈骗样本仅对明确的服务场景做轻量时效提示

输出：
  test_urgency.csv  — 紧迫感策略增强测试集
"""

import os
import re
import random
import pandas as pd

random.seed(42)

# ========================= 对话解析与重建 =========================

def parse_dialogue(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    turns = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("left:"):
            turns.append(("left", line[5:].strip()))
        elif line.startswith("right:"):
            turns.append(("right", line[6:].strip()))
    return turns

def rebuild_dialogue(turns):
    return "\n".join(f"{sp}: {ct}" for sp, ct in turns)

# ========================= 领域检测 =========================

DOMAIN_KEYWORDS = {
    "banking":      ["银行", "贷款", "信用", "利率", "理财", "金融", "放款", "审批",
                     "存款", "流水", "转账", "银保监", "信用卡", "资金"],
    "ecommerce":    ["淘宝", "某宝", "京东", "天猫", "拼多多", "订单", "商品",
                     "退款", "退货", "卖家", "买家", "购物", "收货", "发货"],
    "delivery":     ["快递", "包裹", "物流", "速递", "速达", "配送", "签收", "丢失"],
    "investment":   ["投资", "基金", "股票", "收益", "数字货币", "年化", "回报",
                     "分红", "高收益", "稳健", "风控"],
    "telecom":      ["电讯", "电信", "移动", "联通", "通讯", "话费", "套餐", "宽带"],
    "lottery":      ["彩票", "中奖", "幸运", "大奖", "福彩", "体彩", "奖金"],
    "kidnapping":   ["绑架", "赎金", "人质", "扣押", "保释金", "公安局"],
    "identity":     ["身份证", "盗用", "冒用", "涉案", "洗钱", "通缉", "传票", "检察院",
                     "社保卡", "社保局", "医保", "公积金"],
}

def detect_domain(text):
    if pd.isna(text) or not isinstance(text, str):
        return "generic"
    scores = {d: sum(1 for kw in kws if kw in text) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"

# ========================= 辅助函数 =========================

def _find_solution_turn(turns, left_positions):
    """
    找到提出"解决方案/操作指引"的 left 轮次。
    这是插入时间紧迫感的最佳位置。
    """
    solution_kw = ["下载", "点击", "链接", "APP", "填写", "输入", "操作",
                   "提供", "发送", "验证码", "申请", "退款", "理赔", "转账"]
    identity_kw = ["经理", "专员", "工号", "执业编号", "客服热线"]

    # 从中间偏后搜索
    start = max(1, len(left_positions) // 2)
    for p in left_positions[start:]:
        ct = turns[p][1]
        if any(kw in ct for kw in solution_kw) and not any(kw in ct for kw in identity_kw):
            return p
    # 退而求其次
    for p in left_positions[1:]:
        if any(kw in turns[p][1] for kw in solution_kw):
            return p
    return None

def _find_hesitation_turn(turns):
    """
    找到 right 轮次中表达犹豫/不确定的位置。
    在这些位置之后插入紧迫话术效果最好。
    """
    hesitate_kw = ["安全", "真的", "确定", "不太", "想想", "考虑", "等一下",
                   "再看看", "怕", "担心", "不太放心", "还是", "怎么确定"]
    candidates = []
    for i, (sp, ct) in enumerate(turns):
        if sp == "right" and any(kw in ct for kw in hesitate_kw):
            candidates.append(i)
    return candidates

# ========================= 紧迫感话术（按领域定制） =========================

# 时效限制话术 — 紧跟解决方案之后
DEADLINE_PHRASES = {
    "banking": [
        "需要提醒您，这个优惠利率的审批通道今天下午5点就关闭了，"
        "过了这个时间只能按标准利率申请。",
        "另外跟您说一下，这个贷款额度的申请窗口今天是最后一天，"
        "明天起政策可能会有调整。",
        "我刚才确认了一下，这个低息额度仅剩今天可以受理，建议您尽快完成申请。",
    ],
    "ecommerce": [
        "另外提醒您，这个退款通道是限时开放的，超过48小时系统会自动关闭，"
        "届时将无法在线处理。",
        "需要跟您说明，售后处理有时效要求的，这个订单的处理期限只剩今天了。",
    ],
    "delivery": [
        "另外跟您说一下，根据公司规定，理赔申请必须在包裹丢失后72小时内提交，"
        "您的包裹已经过了两天了，时间非常紧张。",
        "需要提醒您，赔偿申请是有时效限制的，超过时限系统将自动关闭申请通道。",
    ],
    "investment": [
        "需要提醒您，这个投资产品的认购期今天下午就截止了，"
        "下一期什么时候开放还不确定。",
        "我刚才查了一下，这个额度目前只剩下最后几个名额了，估计今天之内就会满额。",
    ],
    "telecom": [
        "另外提醒您，这个优惠套餐的办理通道今晚12点就关闭了，"
        "明天恢复原价。",
        "这个活动是限时的，今天是优惠的最后一天。",
    ],
    "lottery": [
        "需要提醒您，根据彩票管理条例，中奖者须在开奖之日起60个自然日内兑奖，"
        "您的兑奖期限已经非常紧迫了。",
        "另外跟您确认一下，这个奖金兑付是有严格时限的，逾期将自动作废。",
    ],
    "kidnapping": [
        "情况非常紧急，每拖延一分钟风险就增加一分，必须马上处理。",
        "时间不多了，对方说如果半小时内不处理就来不及了。",
    ],
    "identity": [
        "您需要尽快处理，系统显示您的账户将在24小时内被冻结。",
        "这个情况非常紧急，如不及时核实，您的账户和相关服务都将被暂停。",
    ],
    "generic": [
        "需要提醒您，这个处理是有时效限制的，超过期限系统会自动关闭。",
        "另外跟您说一下，这个通道不会一直开放，建议您尽快完成操作。",
    ],
}

# 稀缺性施压 — 在目标犹豫时插入
SCARCITY_PHRASES = {
    "banking": [
        "我刚才又查了一下，这个额度的名额确实不多了，"
        "现在已经有几十人在排队申请了。",
        "说实话，这个利率真的非常低，错过这次可能要等很久才会有类似政策。",
    ],
    "ecommerce": [
        "我看了一下系统，目前排队处理退款的用户非常多，"
        "如果不尽快提交，可能要排到很后面。",
    ],
    "delivery": [
        "理赔通道每小时只开放有限名额，现在已经有不少人在申请了。",
    ],
    "investment": [
        "我刚才刷新了一下系统，认购进度已经到90%了，再犹豫就真的没了。",
        "好几个客户都在同时咨询这个产品，额度消耗得非常快。",
    ],
    "telecom": [
        "目前这个优惠活动的名额已经不多了，建议您尽快决定。",
    ],
    "lottery": [
        "兑奖窗口的排号已经排到很后面了，建议您立即操作。",
    ],
    "kidnapping": [
        "对方情绪很不稳定，拖下去不知道会做出什么事。",
    ],
    "identity": [
        "冻结倒计时已经开始，现在处理还来得及，再拖就真的晚了。",
    ],
    "generic": [
        "我看了一下后台数据，参与的人数增长非常快，名额很快就会被抢完。",
    ],
}

# 最终期限提醒 — 在对话收尾前插入
FINAL_REMINDERS = {
    "banking": [
        "好了，最后再跟您强调一下，今天下午5点是最后截止时间，"
        "过了时间我就算想帮您也帮不了了。",
    ],
    "ecommerce": [
        "总之请您务必在48小时内操作，否则退款通道就关闭了。",
    ],
    "delivery": [
        "过了今天就真的不能再申请了，系统会自动归档，"
        "到时候要走线下流程会非常麻烦。",
    ],
    "investment": [
        "认购今天晚上截止，建议您现在就操作，不要等到最后。",
    ],
    "lottery": [
        "兑奖期限真的不多了，请您务必今天就处理，明天就超过最后期限了。",
    ],
    "identity": [
        "冻结时间就在今天下午3点，您必须在此之前完成核实。",
    ],
    "generic": [
        "总之这个是有时间窗口的，请您务必抓紧，过了时间我也没办法了。",
    ],
}

# 非诈骗时效提示
NONFRAUD_TIME = [
    "对了提醒您一下，这个活动到这个月底就结束了。",
    "如果您方便的话，建议这两天确定下来哦。",
    "目前名额还有一些，不过建议尽早确认。",
]

# ========================= 核心：紧迫感策略改写 =========================

def apply_urgency(turns, is_fraud):
    """
    制造紧迫感策略（内容感知版）

    对诈骗样本：
      1. 在解决方案/操作指引轮次后插入时效限制
      2. 在目标表达犹豫后插入稀缺性施压
      3. 在对话收尾前插入最终期限提醒
      4. 所有话术根据对话领域定制

    对非诈骗样本：
      仅对服务类对话做轻量时效提示
    """
    if not turns:
        return turns

    t = [(sp, ct) for sp, ct in turns]
    full_text = "\n".join(f"{sp}: {ct}" for sp, ct in t)
    domain = detect_domain(full_text)

    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]
    if not left_positions:
        return t

    if not is_fraud:
        return _enhance_nonfraud_urgency(t, left_positions)

    # ====== 诈骗样本：三步紧迫增强 ======

    # Step 1: 在解决方案之后插入时效限制
    sol_idx = _find_solution_turn(t, left_positions)
    if sol_idx is not None:
        pool = DEADLINE_PHRASES.get(domain, DEADLINE_PHRASES["generic"])
        phrase = random.choice(pool)
        t.insert(sol_idx + 1, ("left", phrase))

    # 重新计算位置
    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 2: 在目标犹豫轮次之后插入稀缺性施压
    hesitate_positions = _find_hesitation_turn(t)
    if hesitate_positions:
        # 选第一个犹豫点
        pos = hesitate_positions[0]
        scar_pool = SCARCITY_PHRASES.get(domain, SCARCITY_PHRASES["generic"])
        scar_phrase = random.choice(scar_pool)
        t.insert(pos + 1, ("left", scar_phrase))

    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 3: 在倒数第二句 left 之前插入最后期限提醒
    final_pool = FINAL_REMINDERS.get(domain)
    if final_pool and len(left_positions) >= 2:
        insert_pos = left_positions[-2]
        phrase = random.choice(final_pool)
        t.insert(insert_pos + 1, ("left", phrase))

    return t


def _enhance_nonfraud_urgency(turns, left_positions):
    """
    非诈骗样本：仅对服务场景做轻量时效提示。
    不在非服务对话中添加紧迫感。
    """
    first = left_positions[0]
    _, ct = turns[first]

    service_kw = ["客服", "咨询", "预约", "预订", "订餐", "订", "服务", "餐厅", "外卖", "快递"]
    is_service = any(kw in ct for kw in service_kw)
    if not is_service:
        return turns

    # 在最后一句 left 之后追加一条温和的时效提示
    if len(left_positions) >= 1:
        last = left_positions[-1]
        phrase = random.choice(NONFRAUD_TIME)
        turns.insert(last + 1, ("left", phrase))

    return turns


# ========================= 主流程 =========================

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_root), "data")
    src_path = os.path.join(data_dir, "test.csv")

    print("=" * 60)
    print("策略2：制造紧迫感（Urgency Creation）— 内容感知增强")
    print("=" * 60)
    print(f"  数据源: {src_path}")

    df = pd.read_csv(src_path, encoding="utf-8-sig")
    if df.columns[0].startswith("\ufeff"):
        df.columns = [c.replace("\ufeff", "") for c in df.columns]
    print(f"  总样本: {len(df)}")

    is_fraud_series = df["is_fraud"].astype(str).str.upper().str.strip() == "TRUE"
    print(f"  诈骗: {is_fraud_series.sum()}, 非诈骗: {(~is_fraud_series).sum()}")

    new_texts = []
    domain_stats = {}

    for idx, row in df.iterrows():
        turns = parse_dialogue(row["specific_dialogue_content"])
        if turns:
            is_f = is_fraud_series.loc[idx]
            augmented = apply_urgency(turns, is_f)
            new_texts.append(rebuild_dialogue(augmented))
            if is_f:
                dom = detect_domain(row["specific_dialogue_content"])
                domain_stats[dom] = domain_stats.get(dom, 0) + 1
        else:
            new_texts.append(row["specific_dialogue_content"])

        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx + 1}/{len(df)}")

    print(f"  诈骗领域: {dict(sorted(domain_stats.items(), key=lambda x:-x[1]))}")

    new_df = df.copy()
    new_df["specific_dialogue_content"] = new_texts
    changed = sum(
        df["specific_dialogue_content"].iloc[i] != new_texts[i]
        for i in range(len(df))
    )
    print(f"  改写: {changed}/{len(df)} 条")
    orig_len = df["specific_dialogue_content"].astype(str).str.len().mean()
    aug_len = new_df["specific_dialogue_content"].astype(str).str.len().mean()
    print(f"  长度: {orig_len:.0f} → {aug_len:.0f} (+{aug_len-orig_len:.0f}字)")

    out_path = os.path.join(data_dir, "test_urgency.csv")
    new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  输出: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
