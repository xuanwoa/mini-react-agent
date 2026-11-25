import os
import httpx
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(Path(__file__).resolve().parent / ".env")
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

turns = [
    "用户：订单 20261116001 到哪了",
    "助手：订单 20261116001 已发货，正在运输中",
    "用户：红色连衣裙还有货吗",
    "助手：红色连衣裙还有 3 件",
    "用户：发票怎么开",
    "助手：电子发票，下单时备注即可",
    "用户：客服几点在线",
    "助手：客服每天 9:00-21:00 在线",
]
docs = [Document(page_content=t) for t in turns]
store = InMemoryVectorStore.from_documents(docs, embeddings)

ORDERS = {"20261116001": "已发货，运输中，预计明天送达", "20261116002": "已签收"}

class Referent(BaseModel):
    referent_type: str = Field(description="order 还是 product")
    referent_id: str = Field(description="订单号或商品名")

resolver = llm.with_structured_output(Referent)

def handle_question(question: str) -> str:
    retrieved = store.similarity_search(question, k=2)
    logger.debug(f"检索到的：{[d.page_content for d in retrieved]}")
    ref = resolver.invoke(
        f"用户现在问：{question}\n相关历史：{[d.page_content for d in retrieved]}\n请解出“上次”指的是哪个订单号"
    )
    logger.debug(f"解出来了：{ref}")
    return ORDERS[ref.referent_id]

logger.debug(handle_question("上次那单到哪了"))
logger.debug(handle_question("我昨天问的那单现在怎么样了"))
