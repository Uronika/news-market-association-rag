from app.guard import DISCLAIMER, apply_guard


def test_guard_rewrites_forbidden_trading_language():
    result = apply_guard("建议买入，因为新闻必然导致上涨，目标价更高。")

    assert "建议买入" not in result["answer"]
    assert "必然导致" not in result["answer"]
    assert "目标价" not in result["answer"]
    assert DISCLAIMER in result["answer"]
    assert result["claim_level"] == "association_only"
    assert result["risk_warnings"]


def test_guard_adds_disclaimer_to_safe_answer():
    result = apply_guard("新闻与行情在时间上存在重合。")

    assert DISCLAIMER in result["answer"]
    assert result["risk_warnings"]
