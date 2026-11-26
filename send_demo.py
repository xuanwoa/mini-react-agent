import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from loguru import logger

# 假订单库：一次查 3 个订单
ORDERS = {
    "20261116001": "订单 20261116001：已发货，快递运输中，预计明天送达。",
    "20261116002": "订单 20261116002：已签收。",
    "20261116003": "订单 20261116003：待发货，预计明天发出。",
}


class MapState(TypedDict):
    order_ids: list[str]
    # 并行分支同时往 results 里写，必须用 operator.add 拼起来，不能互相覆盖
    results: Annotated[list, operator.add]


# plan 节点：只宣布要查哪几个订单，不动手
def plan(state):
    logger.debug(f"plan：要查 {len(state['order_ids'])} 个订单")
    return {}


# 路由函数：订单有几个，就发几个 Send；每个 Send 带自己的 order_id
def fan_out(state):
    return [Send("check_one", {"order_id": oid}) for oid in state["order_ids"]]


def check_one(state):
    oid = state["order_id"]
    logger.debug(f"节点 check_one 开始 {time.strftime('%H:%M:%S')} order_id={oid}")
    time.sleep(1)  # 模拟慢接口：每个查单要 1 秒
    result = ORDERS.get(oid, f"订单 {oid}：查询不到，请确认订单号。")
    logger.debug(f"节点 check_one：{result}")
    return {"results": [result]}


# reduce 节点：等 3 个分支都跑完才跑，汇总结果
def reduce_node(state):
    logger.debug(f"汇总：共 {len(state['results'])} 条")
    for r in state["results"]:
        logger.debug(f"  - {r}")
    return {}


builder = StateGraph(MapState)
builder.add_node("plan", plan)
builder.add_node("check_one", check_one)
builder.add_node("reduce", reduce_node)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out)
builder.add_edge("check_one", "reduce")
builder.add_edge("reduce", END)

graph = builder.compile()

start = time.perf_counter()
result = graph.invoke({"order_ids": ["20261116001", "20261116002", "20261116003"], "results": []})
logger.debug(f"总耗时：{time.perf_counter() - start:.2f} 秒（3 个各 1 秒的查单如果串行要 3 秒多）")

print(graph.get_graph().draw_ascii())
