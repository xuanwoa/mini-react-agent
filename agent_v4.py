import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.memory import InMemorySaver
from pydantic import BaseModel, Field
from loguru import logger
import httpx

load_dotenv()
http_client = httpx.Client(proxy="http://127.0.0.1:7890", timeout=60.0)
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
    http_client=http_client,
)
embeddings = OpenAIEmbeddings(
    model=os.getenv("JINA_MODEL_NAME"),
    base_url=os.getenv("JINA_BASE_URL"),
    api_key=os.getenv("JINA_API_KEY"),
    http_client=http_client,
    check_embedding_ctx_length=False,
)

# 店铺资料（agent_v3 那 10 条，照搬）
shop_docs = [
    "退货政策：7 天内无理由退货，运费由买家承担，寄回杭州西湖区仓储中心。",
    "换货政策：7 天内可换货，流程与退货相同。",
    "运费：退货运费由买家承担，换货运费由商家承担。",
    "保修：商品保修 1 年，人为损坏不保修。",
    "发票：可开电子发票，下单时备注即可。",
    "客服时间：每天 9:00-21:00 在线。",
    "发货：每天 16:00 前下单当天发货，16:00 后次日发货。",
    "签收：请当面验货，有问题 24 小时内联系客服。",
    "优惠券：满 100 减 10，新人首单立减 5 元。",
    "售后地址：杭州西湖区仓储中心，电话 0571-88888888。",
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents(shop_docs)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

ORDERS = {
    "20261116001": "已发货，运输中，预计明天送达",
    "20261116002": "已签收，2026-11-15 14:30",
}


@tool
def search_shop_docs(question: str) -> str:
    """查店铺资料：退货、换货、运费、保修、发票、客服时间等问题用这个工具。"""
    docs = retriever.invoke(question)
    return "\n".join(d.page_content for d in docs)


@tool
def check_order(order_id: str) -> str:
    """查订单状态：输入 11 位订单号，返回物流状态。"""
    return ORDERS.get(order_id, "查不到该订单，请确认订单号")


# ★ 关键：先分类，再动手
class RouteDecision(BaseModel):
    needs_order: bool = Field(description="问题里有没有订单号、要不要查订单状态")
    needs_policy: bool = Field(description="问题里有没有问退货、换货、运费、保修这类政策")
    order_id: str | None = Field(default=None, description="问题里的 11 位订单号，没有就填 None")


classifier = llm.with_structured_output(RouteDecision)

agent = create_agent(
    model=llm,
    tools=[search_shop_docs, check_order],
    middleware=[SummarizationMiddleware(model=llm, trigger=("messages", 6), keep=("messages", 2))],
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "1"}}

questions = [
    "买了 5 天能退货吗？",
    "订单 20261116001 到哪了？",
    "订单 20261116001 退了怎么办？",
]

for q in questions:
    # 第 1 步：分类
    route = classifier.invoke(f"判断下面这个问题需要查什么：{q}")
    logger.debug(f"问题：{q}")
    logger.debug(f"分类：查订单={route.needs_order}，查政策={route.needs_policy}，订单号={route.order_id}")

    # 第 2 步：按分类结果，拼一个明确的指令给 agent
    hint = ""
    if route.needs_order and route.needs_policy:
        hint = f"这个问题要同时查订单 {route.order_id} 和退货政策，两个工具都要调。"
    elif route.needs_order:
        hint = f"这个问题只查订单 {route.order_id}。"
    else:
        hint = "这个问题只查店铺资料。"
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"{q}\n（提示：{hint}）"}]},
        config=config,
    )
    logger.debug(f"回答：{result['messages'][-1].content}")
    logger.debug(f"消息数：{len(result['messages'])}")
