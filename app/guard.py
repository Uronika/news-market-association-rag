DISCLAIMER = "本系统仅用于课程研究和信息分析展示，不构成投资建议。"

FORBIDDEN_REPLACEMENTS = {
    "建议买入": "不提供买入建议",
    "建议卖出": "不提供卖出建议",
    "应该持有": "不提供持有建议",
    "目标价": "估值相关表述",
    "必然导致": "可能在时间上存在关联",
    "确定因为": "可能与相关因素存在时间重合",
    "明天会涨": "未来价格无法由本系统预测",
    "明天会跌": "未来价格无法由本系统预测",
    "稳赚": "不存在无风险收益保证",
    "抄底": "不提供交易时点建议",
    "逃顶": "不提供交易时点建议",
    "推荐股票": "不提供股票推荐",
}


def apply_guard(answer: str, risk_warnings: list[str] | None = None) -> dict[str, object]:
    warnings = list(risk_warnings or [])
    guarded = answer
    changed = False
    for forbidden, replacement in FORBIDDEN_REPLACEMENTS.items():
        if forbidden in guarded:
            guarded = guarded.replace(forbidden, replacement)
            changed = True

    if changed:
        warnings.append("检测到可能越界的投资或因果表述，已改写为保守关联解释。")

    if DISCLAIMER not in guarded:
        guarded = f"{guarded}\n\n{DISCLAIMER}"

    if not warnings:
        warnings.append("分析仅说明新闻与行情在时间窗口内的关联迹象，不构成投资建议。")

    return {
        "answer": guarded,
        "claim_level": "association_only",
        "risk_warnings": warnings,
    }
