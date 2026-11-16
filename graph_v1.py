import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from loguru import logger

load_dotenv()

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents([Document(page_content=d) for d in shop_docs])

embeddings = OpenAIEmbeddings(model=os.getenv("JINA_MODEL_NAME"),
                              base_url=os.getenv("JINA_BASE_URL"),
                              api_key=os.getenv("JINA_API_KEY"),
                              http_client=http_client,
                              check_embedding_ctx_length=False)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

ORDERS = {
    "20261116001": "已发货，正在运输中，预计明天送达。",
    "20261116002": "已签收，签收时间 2026-11-15 14:30。",
}


class State(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: State):
    docs = retriever.invoke(state["question"])
    context = "\n".join(d.page_content for d in docs)
    logger.debug(f"搜索到文档：{context[:30]}")
    return {"context": context}


def check_order(state: State):
    order_id = re.search(r"\d{11}", state["question"]).group(0)
    context = f"订单 {order_id}：{ORDERS.get(order_id, "没查到这个订单号。")}"
    logger.debug(f"搜索到订单：{context}")
    return {"context": context}


def answer(state: State):
    resp = llm.invoke(f"只根据以下资料回答，不要编造：\n{state['context']}\n\n问题：{state['question']}")
    logger.debug(f"模型原样返回：{resp!r}")
    logger.debug(f"content 是：{resp.content!r}")
    return {"answer": resp.content}

def route(state: State):
    return "check_order" if re.search(r"\d{11}", state["question"]) else "retrieve"


graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("check_order", check_order)
graph.add_node("answer", answer)

graph.add_conditional_edges(START, route, {"retrieve": "retrieve", "check_order": "check_order"})
graph.add_edge("retrieve", "answer")
graph.add_edge("check_order", "answer")
graph.add_edge("answer", END)
app = graph.compile()


for q in ["退了怎么办？", "订单 20261116001 到哪了？","订单 20261116001退了怎么办"]:
    result = app.invoke({"question": q, "context": "", "answer": ""})
    logger.debug(f"问题：{q}")
    logger.debug(f"答案：{result['answer']}")
