import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from loguru import logger

load_dotenv()


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，比如 23*1.08"""
    return str(eval(expression))


@tool
def get_current_time() -> str:
    """返回当前时间"""
    return datetime.now().strftime("%Y%m%d%H%M%S")


tools_map = {"calculate": calculate, "get_current_time": get_current_time}

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
)

llm_bind_tools = llm.bind_tools([get_current_time, calculate])

messages = [HumanMessage("现在几点？再帮我算 23*1.08 等于多少")]

for i in range(5):
    logger.debug(i)
    resp = llm_bind_tools.invoke(messages)
    if not resp.tool_calls:
        logger.debug(resp.content)
        break
    # A model response can contain more than one tool call. OpenAI requires
    # exactly one ToolMessage for every tool_call_id before the next model call.
    messages.append(resp)
    for call in resp.tool_calls:
        tool_fn = tools_map.get(call["name"])
        if tool_fn is None:
            raise ValueError(f"Unknown tool requested: {call['name']}")

        result = tool_fn.invoke(call["args"])
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
            )
        )
