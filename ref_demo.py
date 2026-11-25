import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
)

# 假装这是数据库里的聊天记录：昨天聊的
history = [
    {"role": "user", "content": "订单 20261116001 到哪了"},
    {"role": "assistant", "content": "订单 20261116001 已发货，正在运输中"},
    {"role": "user", "content": "红色连衣裙还有货吗"},
    {"role": "assistant", "content": "红色连衣裙还有 3 件"},
]

# 假装这是订单库：实时状态
ORDERS = {"20261116001": "已发货，运输中，预计明天送达", "20261116002": "已签收"}

class Referent(BaseModel):
    referent_type: str = Field(description="order 还是 product")
    referent_id: str = Field(description="订单号或商品名")

resolver = llm.with_structured_output(Referent)
ref = resolver.invoke(
    f"用户现在问：上次那单到哪了\n聊天记录：{history}\n请解出“上次那单”指的是哪个订单号"
)
logger.debug(f"解出来了：{ref}")
logger.debug(f"实时状态：{ORDERS[ref.referent_id]}")
