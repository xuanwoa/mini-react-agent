import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
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


@tool
def convert_currency(amount: float, rate: float) -> float:
    """把美元按汇率兑换成人民币，比如 amount=128, rate=7.2"""
    return str(amount * rate)


llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
)

agent = create_agent(
    model=llm,
    tools=[calculate, get_current_time, convert_currency],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "现在几点？再帮我算128美元换成人民币是多少"}]},
    config=config,
)
logger.debug(result["messages"][-1].content)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "那换成欧元呢，按7.8"}]},
    config=config,
)
logger.debug(result2["messages"][-1].content)
