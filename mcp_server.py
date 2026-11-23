from mcp.server.mcpserver import MCPServer

mcp = MCPServer("merchant-assistant")

POLICIES = {
    "退货": "退货政策：下单后 7 天内无理由退货，商品需保持完好，退货运费由买家承担。超期退货：超过 7 天不再受理无理由退货，有质量问题走售后流程。",
    "换货": "换货政策：商品有质量问题，15 天内免费换货，换货运费由商家承担。",
    "运费": "运费：满 99 元包邮，不满 99 元收 8 元运费。",
    "保修": "保修：整机保修 1 年，电池保修 6 个月，人为损坏不在保修范围内。",
    "发票": "发票：下单时可申请电子发票，发票在发货后 3 个工作日内开具。",
    "客服": "客服时间：客服工作时间是每天 9:00-21:00，节假日不休息。",
    "发货": "发货时间：每天 16:00 前下单当天发货，16:00 后下单次日发货。",
}


@mcp.tool()
def check_order(order_id: str) -> str:
    """按订单号查订单状态，比如 20261116001。用户问订单到哪了、发货了没，就用这个工具。"""
    if order_id == "20261116001":
        return "订单 20261116001：已发货，快递运输中，预计明天送达。"
    return "查询不到订单，请确认订单号。"


@mcp.tool()
def search_shop_docs(question: str) -> str:
    """查店铺政策。用户问退货、退了怎么办、怎么退、换货、运费、保修、发票、客服时间、发货时间时，必须用这个工具查政策，不要凭记忆回答。"""
    hits = [v for k, v in POLICIES.items() if k in question or k[0] in question]
    return "\n".join(hits) if hits else "没有找到相关资料，请联系客服"


if __name__ == "__main__":
    mcp.run()
