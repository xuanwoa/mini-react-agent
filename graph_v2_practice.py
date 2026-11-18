# 练习版：照着 graph_v2.py，把 TODO 的地方自己敲一遍
# 不用敲的部分已经填好了（llm、向量库、假查单函数）
# 敲的顺序：State -> classify -> 两个路由函数 -> 其余节点 -> 拼图
import os
import re
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env")
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel, Field

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


def fake_check_order(order_id):
    if order_id == "20261116001":
        return "订单 20261116001：已发货，快递运输中，预计明天送达。"
    return "查询不到订单，请确认订单号。"


# TODO 1：State 是整张图共用的黑板，写全 7 个字段
# question: str / order_id: str / needs_order: bool / needs_policy: bool
# order_result: str / policy_context: str / answer: str
class AgentState(TypedDict):
    pass


# TODO 2：分类结果模型，3 个字段
# needs_order: bool / needs_policy: bool / order_id: str（默认 ""）
class RouteDecision(BaseModel):
    pass


# TODO 3：classify 节点
# 用 llm.with_structured_output(RouteDecision) 判断，返回 dict 写回
# needs_order / needs_policy / order_id
def classify(state):
    raise NotImplementedError("TODO 3：先写 classify 节点")


# TODO 4：check_order 节点，调 fake_check_order，返回 {"order_result": ...}
def check_order_node(state):
    raise NotImplementedError("TODO 4：先写 check_order 节点")


# TODO 5：search_docs 节点，用 retriever 搜，返回 {"policy_context": ...}
def search_docs_node(state):
    raise NotImplementedError("TODO 5：先写 search_docs 节点")


# TODO 6：answer 节点，把 order_result 和 policy_context 拼起来喂给模型
# 只根据资料回答，返回 {"answer": ...}
def answer_node(state):
    raise NotImplementedError("TODO 6：先写 answer 节点")


# TODO 7：路由函数 1——分类完去哪
# 注意：if 的顺序就是优先级，两个都是 True 时先去查订单
def route_after_classify(state):
    raise NotImplementedError("TODO 7：先写 route_after_classify")


# TODO 8：路由函数 2——查完订单去哪
def route_after_order(state):
    raise NotImplementedError("TODO 8：先写 route_after_order")


# TODO 9：拼图——4 个节点、START 边、两条条件边、两条普通边、compile
# builder = StateGraph(AgentState)
# builder.add_node(...) x 4
# builder.add_edge(START, "classify")
# builder.add_conditional_edges("classify", route_after_classify, {...})
# builder.add_conditional_edges("check_order", route_after_order, {...})
# builder.add_edge("search_docs", "answer")
# builder.add_edge("answer", END)
# graph = builder.compile()
raise NotImplementedError("TODO 9：先把图拼出来，再跑下面三道题")

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

print(graph.get_graph().draw_ascii())
