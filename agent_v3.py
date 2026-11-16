import os

import httpx
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.checkpoint import InMemorySaver
from loguru import logger

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
logger.debug(f"切完一共 {len(chunks)} 块")

embeddings = OpenAIEmbeddings(model=os.getenv("JINA_MODEL_NAME"),
                              base_url=os.getenv("JINA_BASE_URL"),
                              api_key=os.getenv("JINA_API_KEY"),
                              http_client=http_client,
                              check_embedding_ctx_length=False)
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def search_shop_docs(question: str) -> str:
    """查店铺资料：退货、换货、运费、保修、发票、客服时间等问题，先用这个工具查。"""
    docs = retriever.invoke(question)
    return "\n".join(d.page_content for d in docs)


ORDERS = {
    "20261116001": "已发货，正在运输中，预计明天送达。",
    "20261116002": "已签收，签收时间 2026-11-15 14:30。",
}


@tool
def check_order(order_id: str) -> str:
    """查订单状态：用户问订单到哪了、发货了没、签收了没，用这个工具，参数是订单号。"""
    return ORDERS.get(order_id, "没查到这个订单号，请确认订单号对不对。")


agent = create_agent(model=llm,
                     tools=[search_shop_docs, check_order],
                     middleware=[SummarizationMiddleware(model=llm,
                                                         trigger=("messages", 6),
                                                         keep=("messages", 2))],
                     checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}

questions = ["退了怎么办？", "订单 20261116001 到哪了？", "保修多久？",
             "运费怎么算？", "客服几点上班？"]
for q in questions:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]}, config)
    logger.debug(f"问完「{q}」，消息数：{len(result['messages'])}")
    logger.debug(result["messages"][-1].content)

# 最后再问一句最开始的，看摘要里还记不记得
result = agent.invoke({"messages": [{"role": "user", "content": "退货是几天内？"}]}, config)
logger.debug(result["messages"][-1].content)
