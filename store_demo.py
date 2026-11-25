from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

load_dotenv(Path(__file__).resolve().parent / ".env")

class State(TypedDict):
    question: str
    order_id: str

def save_order(state: State, runtime: Runtime):
    runtime.store.put(("u1", "memories"), "last_order", {"order_id": "20261116001"})
    logger.debug("已存进 store：20261116001")
    return {"order_id": "20261116001"}

def recall_order(state: State, runtime: Runtime):
    item = runtime.store.get(("u1", "memories"), "last_order")
    order_id = item.value["order_id"] if item else "没记住"
    logger.debug(f"从 store 读出：{order_id}")
    return {"order_id": order_id}

store = InMemoryStore()
checkpointer = InMemorySaver()

save_builder = StateGraph(State)
save_builder.add_node("save_order", save_order)
save_builder.add_edge(START, "save_order")
save_builder.add_edge("save_order", END)
save_graph = save_builder.compile(checkpointer=checkpointer, store=store)

recall_builder = StateGraph(State)
recall_builder.add_node("recall_order", recall_order)
recall_builder.add_edge(START, "recall_order")
recall_builder.add_edge("recall_order", END)
recall_graph = recall_builder.compile(checkpointer=checkpointer, store=store)

# 第一天：thread t1，存进去
r1 = save_graph.invoke({"question": "订单 20261116001 到哪了", "order_id": ""},
                       config={"configurable": {"thread_id": "t1"}})
logger.debug(f"第一天存完：{r1['order_id']}")

# 第二天：换个新 thread t2，不问用户，直接从 store 读
r2 = recall_graph.invoke({"question": "我上次那单到哪了", "order_id": ""},
                         config={"configurable": {"thread_id": "t2"}})
logger.debug(f"第二天读出：{r2['order_id']}")
