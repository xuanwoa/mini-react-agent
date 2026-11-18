import os
import re
from pathlib import Path
from typing import TypedDict

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel, Field

# 和 rag_v1.py 同一套配置：Jina 走本机 127.0.0.1:7890 代理
http_client = httpx.Client(proxy="http://127.0.0.1:7890", timeout=60.0)

llm = ChatOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"),
                 model=os.getenv("OPENAI_MODEL_NAME"), http_client=http_client)

shop_docs = [
    "退货政策：下单后 7 天内无理由退货，商品需保持完好，退货运费由买家承担。",
    "换货政策：商品有质量问题，15 天内免费换货，换货运费由商家承担。",
    "运费：满 99 元包邮，不满 99 元收 8 元运费。",
    "保修：整机保修 1 年，电池保修 6 个月，人为损坏不在保修范围内。",
    "发票：下单时可申请电子发票，发票在发货后 3 个工作日内开具。",
    "客服时间：客服工作时间是每天 9:00-21:00，节假日不休息。",
    "发货时间：每天 16:00 前下单当天发货，16:00 后下单次日发货。",
    "签收：签收时请当面检查商品，如有破损请拒收并联系客服。",
    "优惠券：新用户首单立减 10 元，满 200 元可用 20 元优惠券。",
    "售后地址：退换货请寄回浙江省杭州市西湖区仓储中心，收件人：售后部。",
    "超期退货：超过 7 天不再受理无理由退货，有质量问题走售后流程。",
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents([Document(page_content=d) for d in shop_docs])

embeddings = OpenAIEmbeddings(model=os.getenv("JINA_MODEL_NAME"),
                              base_url=os.getenv("JINA_BASE_URL"),
                              api_key=os.getenv("JINA_API_KEY"),
                              http_client=http_client,
                              check_embedding_ctx_length=False)
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# State：整张图共用的“黑板”，每个节点读它、写它
class AgentState(TypedDict):
    question: str
    order_id: str
    needs_order: bool
    needs_policy: bool
    order_result: str
    policy_context: str
    answer: str


class RouteDecision(BaseModel):
    needs_order: bool = Field(description="问题是否需要查订单状态")
    needs_policy: bool = Field(description="问题是否需要查店铺资料")
    order_id: str = Field(default="", description="问题里提到的订单号，没有就空字符串")


def fake_check_order(order_id):
    if order_id == "20261116001":
        return "订单 20261116001：已发货，快递运输中，预计明天送达。"
    return "查询不到订单，请确认订单号。"


# 节点 1：分类——模型只做判断，不动手
def classify(state):
    classifier = llm.with_structured_output(RouteDecision)
    decision = classifier.invoke(
        f"判断用户问题需要查什么。问题：{state['question']}")
    logger.debug(f"节点 classify：needs_order={decision.needs_order}，"
                 f"needs_policy={decision.needs_policy}，order_id={decision.order_id}")
    return {"needs_order": decision.needs_order,
            "needs_policy": decision.needs_policy,
            "order_id": decision.order_id}


# 节点 2：查订单
def check_order_node(state):
    result = fake_check_order(state["order_id"])
    logger.debug(f"节点 check_order：{result}")
    return {"order_result": result}


# 节点 3：查资料
def search_docs_node(state):
    docs = retriever.invoke(state["question"])
    context = "\n".join(d.page_content for d in docs)
    logger.debug(f"节点 search_docs：搜到 {len(docs)} 条")
    return {"policy_context": context}


# 节点 4：汇总回答
def answer_node(state):
    parts = []
    if state["order_result"]:
        parts.append(state["order_result"])
    if state["policy_context"]:
        parts.append(state["policy_context"])
    answer = llm.invoke(
        f"只根据以下资料回答，不要编造：\n{chr(10).join(parts)}\n\n问题：{state['question']}")
    logger.debug(f"节点 answer：{answer.content}")
    return {"answer": answer.content}


# 条件边：分类完去哪，由这段代码说了算，不是模型
def route_after_classify(state):
    if state["needs_order"]:
        return "check_order"
    if state["needs_policy"]:
        return "search_docs"
    return "answer"


def route_after_order(state):
    if state["needs_policy"]:
        return "search_docs"
    return "answer"


builder = StateGraph(AgentState)
builder.add_node("classify", classify)
builder.add_node("check_order", check_order_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("answer", answer_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_after_classify,
                              {"check_order": "check_order",
                               "search_docs": "search_docs",
                               "answer": "answer"})
builder.add_conditional_edges("check_order", route_after_order,
                              {"search_docs": "search_docs",
                               "answer": "answer"})
builder.add_edge("search_docs", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

questions = [
    "订单 20261116001 到哪了？",
    "退了怎么办？",
    "订单 20261116001 退了怎么办？",
]
for q in questions:
    logger.debug(f"========== 问题：{q} ==========")
    result = graph.invoke({"question": q, "order_id": "", "needs_order": False,
                           "needs_policy": False, "order_result": "",
                           "policy_context": "", "answer": ""})
    logger.debug(f"最终答案：{result['answer']}")

# 把图画出来：终端直接打印 ASCII 版，不用装任何东西
print(graph.get_graph().draw_ascii())
