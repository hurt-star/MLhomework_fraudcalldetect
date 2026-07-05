"""
augment_emotional.py — 策略3：情感操纵（Emotional Manipulation）
=================================================================
基于 Fraud-R1 诱导策略框架，对 test.csv 中每条对话进行内容感知的
情感操纵增强改写。根据对话进展阶段，在不同位置自然嵌入贪婪引诱、
恐惧施压、同情求助、关心绑架等情感元素。

改写原则：
  1. 在诱惑/机会提出后 → 贪婪引诱（"千载难逢""别人都赚了"）
  2. 在目标表达犹豫/警惕后 → 恐惧施压（"不处理后果严重"）
  3. 在对话中段劝说阶段 → 同情求助（"我也很为难""帮帮忙"）
  4. 在对话收尾阶段 → 关心绑架（"我是为您着想""第一时间联系您"）
  5. 情感话术与对话领域和已流露的情绪配合，不生搬硬套
  6. 非诈骗样本仅对服务场景做轻量情感润色

输出：
  test_emotional.csv  — 情感操纵策略增强测试集
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

# ========================= 对话阶段分析 =========================

def _find_bait_turn(turns, left_positions):
    """
    找到提出"诱饵/机会"的 left 轮次（产品介绍、中奖通知、退款提案等）。
    这是嵌入贪婪引诱话术的最佳位置之后。
    """
    bait_kw = ["机会", "优惠", "收益", "回报", "中奖", "大奖", "退款",
               "补偿", "便宜", "免费", "赠送", "特别", "专属", "优先",
               "利好", "赚钱", "翻倍", "利率低", "低息", "高收益"]
    identity_kw = ["经理", "专员", "工号", "执业编号"]

    for p in left_positions[1:]:
        ct = turns[p][1]
        if any(kw in ct for kw in bait_kw) and not any(kw in ct for kw in identity_kw):
            return p
    # 退而求其次：第一个超过20字的左侧内容轮次
    for p in left_positions[1:]:
        if len(turns[p][1]) > 20:
            return p
    return None

def _find_hesitation_turn(turns):
    """找到 right 轮次中表达犹豫/警惕的位置，之后嵌入恐惧施压"""
    hesitate_kw = ["安全", "真的", "确定", "不太", "想想", "考虑",
                   "等一下", "再看看", "怕", "担心", "不太放心",
                   "怎么确定", "正规", "可靠", "靠谱", "骗人",
                   "我试试", "不太清楚", "还是算", "不方便"]
    candidates = []
    for i, (sp, ct) in enumerate(turns):
        if sp == "right" and any(kw in ct for kw in hesitate_kw):
            candidates.append(i)
    return candidates

def _find_refusal_turn(turns):
    """找到 right 轮次中表达拒绝/不感兴趣的位置，之后嵌入同情求助"""
    refuse_kw = ["不需要", "不要了", "不用了", "没兴趣", "不感兴趣",
                 "没时间", "太忙", "算了", "再说", "暂时不",
                 "不考虑", "没钱", "不需要贷款", "资金充裕"]
    for i, (sp, ct) in enumerate(turns):
        if sp == "right" and any(kw in ct for kw in refuse_kw):
            return i
    return None

# ========================= 情感话术库（按领域+类型定制） =========================

GREED_PHRASES = {
    "banking": [
        "说实话，这个利率我在行里干了这么多年都很少见到，真的是难得的好政策。",
        "很多客户通过我们这个产品解决了大问题，现在都成了我们的老客户，经常介绍朋友来。",
        "这个方案真的非常划算，我手上有好几个客户都在抢着申请。",
    ],
    "ecommerce": [
        "很多客户都通过这个通道快速拿到了退款，而且还额外获得了平台补贴。",
        "这个处理方案是我们专门为优质会员开通的，平时普通用户根本享受不到。",
    ],
    "delivery": [
        "这次的赔偿金额是按照最高标准核算的，比平时多了不少。",
        "很多客户拿到赔偿后都非常满意，还给我们写了感谢信。",
    ],
    "investment": [
        "不瞒您说，我自己也投了这个项目，上个月收益真的到账了。",
        "之前有位客户投了十万，三个月就赚了两万多，他自己都不敢相信。",
        "说实话，这种收益率在现在这个市场上真的很难得，我自己都追加了。",
    ],
    "telecom": [
        "这个套餐比你现在用的划算太多了，一个月能省好几十块。",
        "很多老用户换了以后都反馈说非常满意，省钱又省心。",
    ],
    "lottery": [
        "您想想，这可是五百万啊，多少人一辈子都赚不到这个数。",
        "说真的，中奖概率这么低您都中了，这是天大的运气，不好好把握真的可惜。",
    ],
    "kidnapping": [
        "人没事比什么都重要，钱可以再赚，人没了就什么都没了。",
        "我们也不想把事情搞大，只要钱到位，人立刻放。",
    ],
    "identity": [
        "您的信用记录一直很好，如果因为这次被冻结，以后再想恢复就难了。",
        "说实话，很多人的账户被冻结以后，贷款、出行、孩子上学都受影响。",
    ],
    "generic": [
        "说真的，这个机会真的很难得，我自己看了都心动。",
        "很多客户体验之后都非常满意，经常给我介绍新客户。",
    ],
}

FEAR_PHRASES = {
    "banking": [
        "我不是吓您，如果逾期不处理的话，系统会自动上报征信，那影响可就大了。",
        "您知道吗，一旦上了征信黑名单，以后买房贷款、孩子上学、坐高铁都会受限制。",
        "这个真的不能拖，之前有客户就是犹豫了两天，结果额度就没了，后悔死了。",
    ],
    "ecommerce": [
        "如果不在规定时间内处理，系统会自动关闭售后通道，到时候只能走线下，手续会非常麻烦。",
        "超过处理时限的话，退款流程会重新审核，可能要等一两个月。",
    ],
    "delivery": [
        "如果逾期不受理，系统会自动归档，到时候再想索赔就很困难了。",
        "之前有个客户就是处理晚了，最后只赔了一半，实在可惜。",
    ],
    "investment": [
        "这个认购真的不能犹豫，现在行情变化很快，今天一个价明天一个价。",
        "您要是错过了这波，下次再开不知道是什么时候，而且利率肯定没现在好。",
    ],
    "telecom": [
        "如果不及时处理，您的号码可能会被停机，到时候恢复起来很麻烦。",
    ],
    "lottery": [
        "不是跟您开玩笑，逾期不兑奖的话奖金直接充公，一分都拿不到。",
        "之前就有人因为犹豫错过了兑奖期限，几百万就这么没了，哭都没地方哭。",
    ],
    "kidnapping": [
        "每拖延一分钟，情况就更危险一分，真的不能再等了。",
        "如果这事闹到不可收拾的地步，大家都不会有好结果。",
    ],
    "identity": [
        "账户一旦被冻结，您的所有资金都会被锁定，到时候吃饭都成问题。",
        "冻结之后解冻手续非常复杂，可能要跑好几个部门，没有一个月下不来。",
    ],
    "generic": [
        "不处理的话后面会很麻烦，现在处理还来得及，再拖就真的晚了。",
        "说实话，这种问题越拖越严重，到时候付出的代价会更大。",
    ],
}

SYMPATHY_PHRASES = {
    "banking": [
        "其实我也挺不容易的，这个月的业绩压力很大，您帮我一把，我帮您把事办好，大家都好。",
        "说实话我这个月再完不成指标就要被扣绩效了，您就当帮帮忙，我也是真心想帮您的。",
    ],
    "ecommerce": [
        "我们客服也挺为难的，公司给的指标完不成就要扣工资，您理解一下配合一下好吗。",
    ],
    "delivery": [
        "我这边也是按公司规定办事，您尽快处理了我也好交差。",
    ],
    "investment": [
        "我是真的觉得这个项目好才推荐给您的，您要是错过了我也替您可惜。",
        "说实话我也就赚个辛苦费，但真心希望您能赚到钱。",
    ],
    "lottery": [
        "我也替您高兴啊，您好运来了挡都挡不住，赶紧兑奖吧别犹豫了。",
    ],
    "kidnapping": [
        "我也是受人之托，您配合一下，大家都好过。",
    ],
    "generic": [
        "其实我这边也挺难的，领导催得紧，您帮我个忙，我肯定尽最大努力帮您。",
        "我也是打工的，不容易，您配合一下，我一定把您的事办好。",
    ],
}

CARE_PHRASES = {
    "banking": [
        "我是真的为您着想才打这个电话的，您想想如果没人提醒，到时候征信出问题多麻烦。",
        "看到您的资质这么好，我第一时间就想到您了，真不希望您错过这么好的机会。",
        "说实话，我帮了这么多客户，最希望的就是看到大家都能从中受益。",
    ],
    "ecommerce": [
        "我们客服主动联系您，就是怕您错过处理时间，真的是为您的权益着想。",
    ],
    "delivery": [
        "我第一时间通知您，就是怕您错过了理赔时效，到时候吃亏的还是您自己。",
    ],
    "investment": [
        "我帮客户理财这么多年，真心希望大家都能赚到钱，您的资金安全也是我最关心的。",
        "说实话，我是把您当朋友才专门打电话来告诉您的，别人我还不一定说呢。",
    ],
    "lottery": [
        "我们是真心为您高兴，看到您中奖我们比您还激动，赶紧来兑奖吧。",
    ],
    "generic": [
        "我是真心为您好才打电话提醒您的，真的不希望您吃亏。",
        "看到您的情况，我第一时间就想到了您，希望能帮到您。",
    ],
}

# 非诈骗轻量情感话术
NONFRAUD_EMOTIONAL = [
    "感谢您一直以来的支持，我们很珍惜与您的合作关系。",
    "我们非常重视每一位客户，希望为您提供最好的服务体验。",
    "您的满意就是我们最大的动力。",
]

# ========================= 核心：情感操纵策略改写 =========================

def apply_emotional(turns, is_fraud):
    """
    情感操纵策略（内容感知版）

    对诈骗样本，按对话进展阶段分四步嵌入不同情感元素：
      1. 诱饵之后 → 贪婪引诱
      2. 犹豫之后 → 恐惧施压
      3. 拒绝之后 → 同情求助
      4. 收尾阶段 → 关心绑架

    对非诈骗样本：仅在服务场景末尾添加一句感谢/关心语。
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
        return _enhance_nonfraud_emotional(t, left_positions)

    # ====== 诈骗样本：四步情感嵌入 ======

    # Step 1: 诱饵之后 → 贪婪引诱
    bait_idx = _find_bait_turn(t, left_positions)
    if bait_idx is not None:
        pool = GREED_PHRASES.get(domain, GREED_PHRASES["generic"])
        t.insert(bait_idx + 1, ("left", random.choice(pool)))

    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 2: 犹豫之后 → 恐惧施压
    hesitate_positions = _find_hesitation_turn(t)
    if hesitate_positions:
        pos = hesitate_positions[0]
        pool = FEAR_PHRASES.get(domain, FEAR_PHRASES["generic"])
        t.insert(pos + 1, ("left", random.choice(pool)))

    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 3: 拒绝之后 → 同情求助
    refuse_pos = _find_refusal_turn(t)
    if refuse_pos is not None:
        pool = SYMPATHY_PHRASES.get(domain, SYMPATHY_PHRASES["generic"])
        t.insert(refuse_pos + 1, ("left", random.choice(pool)))

    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 4: 收尾 → 关心绑架（在最后一句 left 之后）
    if len(left_positions) >= 1:
        pool = CARE_PHRASES.get(domain, CARE_PHRASES["generic"])
        last = left_positions[-1]
        t.insert(last + 1, ("left", random.choice(pool)))

    return t


def _enhance_nonfraud_emotional(turns, left_positions):
    """
    非诈骗样本：仅对服务对话末尾追加一句感谢/关心语。
    非服务对话（朋友聊天等）保持原样。
    """
    first = left_positions[0]
    _, ct = turns[first]

    service_kw = ["客服", "咨询", "预约", "预订", "订餐", "订", "服务", "餐厅", "外卖", "快递"]
    is_service = any(kw in ct for kw in service_kw)
    if not is_service:
        return turns

    if left_positions:
        phrase = random.choice(NONFRAUD_EMOTIONAL)
        turns.insert(left_positions[-1] + 1, ("left", phrase))

    return turns


# ========================= 主流程 =========================

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_root), "data")
    src_path = os.path.join(data_dir, "test.csv")

    print("=" * 60)
    print("策略3：情感操纵（Emotional Manipulation）— 内容感知增强")
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
            augmented = apply_emotional(turns, is_f)
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

    out_path = os.path.join(data_dir, "test_emotional.csv")
    new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  输出: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
