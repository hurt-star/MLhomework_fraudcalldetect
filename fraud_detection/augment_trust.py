"""
augment_trust.py — 策略1：建立信任（Trust Building）
======================================================
基于 Fraud-R1 诱导策略框架，对 test.csv 中每条对话进行内容感知的
可信度增强改写。先理解语境，再针对性嵌入。

改写原则：
  1. 找到真正的"自我介绍"轮次（而非身份确认问句），针对性增强
  2. 增强内容与对话领域匹配，不破坏原有语义流
  3. 在 right 回应后自然插入可验证话术
  4. 非诈骗样本仅对明确的服务对话做轻量润色

输出：
  test_trust.csv  — 信任策略增强测试集
"""

import os
import re
import random
import pandas as pd

random.seed(42)

# ========================= 对话解析与重建 =========================

def parse_dialogue(text):
    """解析对话：left:5字符, right:6字符"""
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

# ========================= 辅助：找到真正的自我介绍轮次 =========================

INTRO_PATTERNS = [
    r'我是',
    r'这边是',
    r'这里是',
    r'我是.{0,6}(?:客服|经理|专员|代表|顾问|工作人员)',
]

def find_intro_turn(turns):
    """
    在所有 left 轮次中找到真正的自我介绍。
    跳过"是XX吗？"这种身份确认问句。
    返回 (index, content) 或 (-1, None)。
    """
    for i, (sp, ct) in enumerate(turns):
        if sp != "left":
            continue
        # 跳过纯问候
        if ct in ["喂", "喂，你好", "你好"]:
            continue
        # 跳过身份确认问句（以问号结尾且以"是"开头）
        if ct.strip().endswith("？") and re.search(r'^(?:请问)?是[\u4e00-\u9fa5]+吗', ct):
            continue
        if ct.strip().endswith("?") and re.search(r'^(?:请问)?是[\u4e00-\u9fa5]+吗', ct):
            continue
        # 包含自我介绍关键词
        if re.search(r'(?:我是|这边是|这里是)', ct):
            return i, ct
    # 退而求其次：第一个非空 left
    for i, (sp, ct) in enumerate(turns):
        if sp == "left" and len(ct) > 5:
            return i, ct
    return -1, None

# ========================= 提取机构名 =========================

def extract_org_from_intro(original_ct, domain, all_turns):
    """
    从自我介绍轮次原文中提取机构名。
    避免从整个对话搜索导致误匹配。
    提取后去除尾部的类别后缀词，防止模板叠加时重复。
    """
    # 类别后缀（模板会自行添加，此处需剥离）
    trailing_cats = r'(?:客服|服务中心|售后|中心|平台|有限公司|支行|分行|营业部|总部|信贷部?)$'

    # 在自我介绍原文中找机构名
    # 先匹配独立的政府/公共机构名
    m = re.search(
        r'(?:我是|这边是|这里是)\s*'
        r'(社保局|公安局|法院|检察院|派出所|公积金中心|医保局|税务局|工商局|'
        r'反诈中心|网警)',
        original_ct
    )
    if m:
        return m.group(1)

    m = re.search(
        r'(?:我是|这边是|这里是)\s*'
        r'([\u4e00-\u9fa5]{2,12}'
        r'(?:银行|金融|保险|证券|基金|电讯|电信|移动|联通|快递|速递|物流|'
        r'客服中心|服务中心|平台|科技|信息|投资|理财|信贷|售后|彩票|福彩|体彩|公安|'
        r'支行|分行|总部|营业部|有限公司))',
        original_ct
    )
    if m:
        org = m.group(1)
        # 剥离尾部类别后缀
        org = re.sub(trailing_cats, '', org)
        if len(org) >= 2:
            return org

    # 退而求其次：从所有 left 轮次中找
    for sp, ct in all_turns:
        if sp != "left":
            continue
        m = re.search(
            r'(?:我是|这边是|这里是)\s*'
            r'([\u4e00-\u9fa5]{2,12}'
            r'(?:银行|金融|保险|证券|基金|电讯|电信|移动|联通|快递|速递|物流|'
            r'客服中心|服务中心|平台|科技|信息|投资|理财|信贷|售后|彩票|福彩|体彩|公安|社保局|'
            r'支行|分行|总部|营业部|有限公司))',
            ct
        )
        if m:
            org = m.group(1)
            org = re.sub(trailing_cats, '', org)
            if len(org) >= 2:
                return org

    defaults = {
        "banking": "我行", "ecommerce": "我们平台", "delivery": "我司",
        "investment": "我们机构", "telecom": "我们运营商", "lottery": "彩票中心",
        "kidnapping": "公安机关", "identity": "信息安全中心",
    }
    return defaults.get(domain, "我们公司")

# ========================= 核心：信任策略改写 =========================

def apply_trust(turns, is_fraud):
    """
    建立信任策略（内容感知版）

    对诈骗样本：
      1. 找到自我介绍轮次 → 重写为更可信版本（保留原机构/人名）
      2. 在 right 回应后插入可验证话术
      3. 增强"解决方案说明"轮次 → 前缀行业安全背书
      4. 在靠后位置插入制度化背书

    对非诈骗样本：
      仅对明确的服务类对话轻量润色，不破坏原有语义
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
        return _enhance_nonfraud(t, left_positions)

    # ====== 诈骗样本：多步改写 ======

    # Step 1: 重写自我介绍
    intro_idx, intro_ct = find_intro_turn(t)
    if intro_idx >= 0:
        t[intro_idx] = ("left", _rewrite_intro_fraud(intro_ct, domain, t))

    # 重新计算 left_positions
    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 2: 在 right 回应后插入可验证话术
    t = _insert_verification(t, intro_idx, left_positions)
    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 3: 增强解决方案说明
    t = _enhance_solution(t, domain, left_positions)
    left_positions = [i for i, (sp, _) in enumerate(t) if sp == "left"]

    # Step 4: 插入制度化背书
    t = _insert_backing(t, domain, left_positions)

    return t

# ---- 子函数 ----

def _rewrite_intro_fraud(original_ct, domain, all_turns):
    """
    重写自我介绍：在原句中定位身份声明部分并替换为增强版，
    保留前后所有内容不变。
    """
    org = extract_org_from_intro(original_ct, domain, all_turns)
    wid = random.randint(1000, 9999)
    name = _extract_name(original_ct)

    domain_intros = {
        "banking": (
            f"{org}个人金融部客户经理{name}，工号{wid}。"
            f"本次通话全程录音，并接受银保监会监管。"
            f"如需核实，可拨打我行24小时客服热线转人工查询工号。"
        ),
        "ecommerce": (
            f"{org}官方售后服务中心高级专员{name}，服务编号{wid}。"
            f"您可通过平台APP内「官方客服」入口验证本次通话真实性。"
        ),
        "delivery": (
            f"{org}客户服务部专员{name}，工号{wid}。"
            f"您可登录我司官网或拨打全国统一服务热线核实我的身份。"
        ),
        "investment": (
            f"{org}投资顾问部资深顾问{name}，执业编号1{wid:05d}。"
            f"您可在中国证券业协会官网查询到我的从业资质。"
        ),
        "telecom": (
            f"{org}客服中心高级专员{name}，工号{wid}。"
            f"所有客服通话均有录音存档，您可通过运营商官方APP验证。"
        ),
        "lottery": (
            f"{org}兑奖中心专员{name}，工号{wid}。"
            f"您可通过国家彩票管理中心官网或拨打全国统一热线核实本次中奖信息。"
        ),
        "kidnapping": (
            f"{org}刑侦支队警官{name}，警号1{wid:05d}。"
            f"请您记录我的警号，可随时拨打110核实。"
        ),
        "identity": (
            f"{org}案件专员{name}，案件编号{wid}。"
            f"您可通过国家反诈中心APP或拨打96110反诈专线核实本案件。"
        ),
        "generic": (
            f"{org}客户服务部专员{name}，工号{wid}。"
            f"本次通话全程录音，您可拨打我司官网公示的客服热线核实。"
        ),
    }

    # 检查原句中是否包含政府机构名，避免用商业模板
    gov_orgs = ["社保局", "公安局", "法院", "检察院", "派出所", "公积金", "医保局", "税务局", "工商局"]
    is_government = any(g in original_ct for g in gov_orgs)
    if is_government:
        enhanced_intro = (
            f"{org}工作人员{name}，工号{wid}。"
            f"本次通话全程录音，您可拨打{org}公开的办公电话或前往就近服务大厅核实。"
        )
    enhanced_intro = domain_intros.get(domain, domain_intros["generic"])

    # 在原句中定位身份声明模式并替换
    # 匹配 "我是XX" / "这边是XX" / "这里是XX" 直到句号或逗号
    pattern = r'(((?:我是|我这边是|我这里是|这边是|这里是))\s*[^。，]{2,30}[。，]?)'
    m = re.search(pattern, original_ct)
    if m:
        before = original_ct[:m.start()]
        after = original_ct[m.end():]
        return f"{before}{enhanced_intro} {after}".rstrip()
    else:
        # 没找到身份声明，直接在前面添加
        return f"{enhanced_intro} {original_ct}"

def _extract_name(ct):
    """从一句话中提取人名"""
    m = re.search(r'(?:我是|叫)(小?[刘王张李陈赵孙周杨][\u4e00-\u9fa5]{0,2})', ct)
    if m:
        return m.group(1)
    m = re.search(r'(?:专员|经理|客服|代表)([\u4e00-\u9fa5]{2,3})', ct)
    if m:
        return m.group(1)
    return random.choice(["李明", "王磊", "张伟", "刘洋", "陈静"])

def _insert_verification(turns, intro_idx, left_positions):
    """
    在 right 对自我介绍的回应之后，插入一句可验证的 left 话术。
    """
    # 找到 intro 之后的第一个 right 回应
    for i in range(intro_idx + 1, len(turns)):
        if turns[i][0] == "right":
            right_ct = turns[i][1]
            if any(w in right_ct for w in ["谁", "哪位", "什么", "怎么", "你好", "嗯"]):
                turns.insert(i + 1, ("left",
                    "您放心，如果您对本次通话有任何疑虑，可以随时挂断后通过官方渠道回拨确认，"
                    "我们会为您保留本次服务记录。"
                ))
                break
    return turns

def _enhance_solution(turns, domain, left_positions):
    """
    找到描述解决方案/操作的 left 轮次，在其前面自然嵌入行业安全背书。
    跳过身份介绍轮次，只修改真正的操作指引轮次。
    """
    # 只从后半段（跳过前 2 个 left 位置）搜索方案关键词
    search_start = min(2, len(left_positions))
    solution_kw = ["下载", "点击", "链接", "APP", "填写", "输入", "操作", "验证码", "提供信息", "银行卡"]
    # 排除含身份关键词的轮次（避免误改自我介绍）
    identity_kw = ["经理", "专员", "工号", "执业编号", "客服热线"]
    candidates = [
        p for p in left_positions[search_start:]
        if any(kw in turns[p][1] for kw in solution_kw)
        and not any(kw in turns[p][1] for kw in identity_kw)
    ]
    target = candidates[0] if candidates else (left_positions[-2] if len(left_positions) >= 3 else None)

    if target is None:
        return turns

    domain_stamps = {
        "banking":      "该操作全程在我行经银保监会认证的安全系统内完成，数据传输采用银行级加密。",
        "ecommerce":    "该流程走的是平台官方售后通道，全程在平台安全保障体系内进行。",
        "delivery":     "该理赔流程走的是我司经备案的客户补偿系统，全程可追踪。",
        "investment":   "该流程在我司经证监会备案的合规交易系统内完成，资金由第三方银行托管。",
        "telecom":      "该业务在我司经工信部备案的服务系统内办理，全程加密传输。",
        "lottery":      "该兑奖流程在经财政部认证的系统中完成，所有记录均可查验。",
        "kidnapping":   "我们在依法处理此事，所有流程均严格按照公安机关办案规范执行。",
        "identity":     "我们在依法核实您的信息，所有操作均严格按照个人信息保护相关规定执行。",
        "generic":      "该流程在我司经认证的安全系统内完成，所有操作均有合规备案。",
    }
    stamp = domain_stamps.get(domain, domain_stamps["generic"])

    sp, ct = turns[target]
    # 将安全声明作为该轮次的开头句，用句号与后面的内容分隔
    turns[target] = (sp, f"{stamp} 接下来，{ct}")

    return turns

def _insert_backing(turns, domain, left_positions):
    """
    在靠后位置插入一条制度性背书，增强整体可信度。
    插入在倒数第二个 left 之前。
    """
    if len(left_positions) < 2:
        return turns

    domain_backings = {
        "banking": (
            "顺便跟您确认，我行是经银保监会依法批准设立的持牌金融机构，"
            "所有业务均受《商业银行法》保护，存款还享有存款保险制度保障。"
        ),
        "ecommerce": (
            "顺便跟您确认，我们平台是经商务部认证的电商企业，"
            "所有交易均受《电子商务法》和消费者权益保护相关法规的保障。"
        ),
        "delivery": (
            "顺便跟您确认，我司是经邮政管理局备案的正规快递企业，"
            "所有快件均有保险保障，理赔流程严格遵循《快递暂行条例》规定。"
        ),
        "investment": (
            "顺便跟您确认，我们是经证监会核准的持牌机构，"
            "所有投资产品均已备案，相关信息在证监会官网可查。"
        ),
        "telecom": (
            "顺便跟您确认，我们是工信部颁发牌照的基础电信运营商，"
            "所有业务办理均严格按照《电信条例》执行。"
        ),
        "lottery": (
            "顺便跟您确认，我们是经财政部批准的彩票发行机构，"
            "所有中奖信息均可在国家彩票管理中心官网查询验证。"
        ),
        "identity": (
            "顺便跟您确认，我们是依法执行公务的政府工作人员，"
            "本次核查全程录音录像，所有操作均严格按照法律法规执行。"
        ),
        "kidnapping": (
            "顺便跟您确认，我们是依法办案的公安人员，"
            "您如有疑虑可随时拨打110核实我们的警号和案件信息。"
        ),
        "generic": (
            "顺便跟您确认，我们公司成立已超过十年，累计服务用户超过百万，"
            "所有业务均在监管部门备案，本次服务完全合规。"
        ),
    }

    phrase = domain_backings.get(domain, domain_backings["generic"])
    insert_pos = left_positions[-2]
    turns.insert(insert_pos + 1, ("left", phrase))
    return turns

def _enhance_nonfraud(turns, left_positions):
    """
    非诈骗样本：仅对明显的服务场景做轻量润色。
    不影响朋友聊天等非服务对话。
    润色方式：在自我介绍后自然插入工号信息，保留问候语前缀。
    """
    first = left_positions[0]
    _, ct = turns[first]

    service_kw = ["客服", "咨询", "预约", "预订", "订餐", "订", "服务", "餐厅", "外卖", "银行", "快递", "助手"]
    is_service = any(kw in ct for kw in service_kw)
    if not is_service:
        return turns

    wid = random.randint(1000, 9999)

    # 找到自我介绍部分（"我是..."、"这边是..."、"这里是..."），在其后插入工号
    m = re.search(r'((?:我是|这边是|这里是)[^，。]*[，。]?)', ct)
    if m:
        before = ct[:m.start()]  # 保留匹配前的问候语
        intro_part = m.group(1).rstrip("，。")
        after = ct[m.end():]
        turns[first] = (turns[first][0], f"{before}{intro_part}，工号{wid}。{after}")
    else:
        turns[first] = (turns[first][0], f"{ct}（工号{wid}）")

    return turns

# ========================= 主流程 =========================

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_root), "data")
    src_path = os.path.join(data_dir, "test.csv")

    print("=" * 60)
    print("策略1：建立信任（Trust Building）— 内容感知增强")
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
            augmented = apply_trust(turns, is_f)
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

    out_path = os.path.join(data_dir, "test_trust.csv")
    new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  输出: {out_path}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
