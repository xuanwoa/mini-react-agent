import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
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
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("messages", 6),
            keep=("messages", 2),
        ),
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}


def show_state(result, label):
    msgs = result["messages"]
    logger.debug(f"--- {label}：消息数 {len(msgs)} ---")
    for m in msgs:
        logger.debug(f"[{m.type}] {str(m.content)[:80]}")


questions = [
    "现在几点？再帮我算 128 美元按 7.2 换成人民币是多少",
    "那换成欧元呢，按 7.8",
    "再算 50 美元按 7.1 换成人民币是多少",
    "再算 200 美元按 7.3 换成人民币是多少",
    "我一开始算的 128 美元换人民币是多少？",
]

for i, q in enumerate(questions, 1):
    result = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
    logger.debug(f"第 {i} 轮回答：{result['messages'][-1].content}")
    show_state(result, f"第 {i} 轮后")
