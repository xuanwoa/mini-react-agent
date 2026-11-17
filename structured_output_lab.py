import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from loguru import logger

load_dotenv()
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
)


class OrderResult(BaseModel):
    order_id: str = Field(description="11 位数字订单号", pattern=r"^\d{11}$")
    status: Literal["已发货", "运输中", "已签收", "查不到"] = Field(description="订单状态，只能是这四个值之一")


# 默认方式（function_calling）
default_llm = llm.with_structured_output(OrderResult)
result1 = default_llm.invoke("订单 20261116001 到哪了？")
logger.debug(f"默认方式：{result1}")

# 严格模式（json_schema）
strict_llm = llm.with_structured_output(OrderResult, method="json_schema")
result2 = strict_llm.invoke("订单 20261116001 到哪了？")
logger.debug(f"严格模式：{result2}")

# include_raw 三件套
raw_llm = llm.with_structured_output(OrderResult, include_raw=True)
result3 = raw_llm.invoke("你好，我想问问退货的事")
logger.debug(f"parsed：{result3['parsed']}")
logger.debug(f"raw：{result3['raw']}")
logger.debug(f"parsing_error：{result3['parsing_error']}")

# parsed 是 None 就填默认值，别让程序崩
if result3["parsed"] is None:
    fallback = OrderResult(order_id="00000000000", status="查不到")
    logger.debug(f"走默认值：{fallback}")
else:
    logger.debug(f"解析成功，不用默认值：{result3['parsed']}")

# 直接喂错 JSON，看 parsing_error 长什么样
parser = PydanticOutputParser(pydantic_object=OrderResult)
try:
    parser.parse('{"order_id": "abc", "status": "飞走了"}')
except Exception as e:
    logger.debug(f"解析失败：{e}")
