from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

load_dotenv(Path(__file__).resolve().parent / ".env")

class State(TypedDict):
    order_id: str
    amount: float
    approved: bool

def decide(state: State):
    if state["amount"] > 500:
        ok = interrupt("退款超过 500 元，需要人工确认，同意吗？")
        return {"approved": ok == "yes"}
    return {"approved": True}

def refund(state: State):
    if state["approved"]:
        logger.debug(f"订单 {state['order_id']} 退款 {state['amount']} 元，已执行")
    else:
        logger.debug(f"订单 {state['order_id']} 退款被人工拒绝")
    return {}

builder = StateGraph(State)
builder.add_node("decide", decide)
builder.add_node("refund", refund)
builder.add_edge(START, "decide")
builder.add_edge("decide", "refund")
builder.add_edge("refund", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}

result = graph.invoke({"order_id": "20261116001", "amount": 800}, config=config)
logger.debug(f"第一次运行停在：{result}")
result = graph.invoke(Command(resume="yes"), config=config)
logger.debug(f"恢复后：{result}")
