import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

load_dotenv(Path(__file__).resolve().parent / ".env")
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
)

async def main():
    # 字符串 = 远程地址：MCPAdapter 不会自己启动 server，server 必须已经在跑
    async with MCPAdapter("http://127.0.0.1:8000/mcp") as adapter:
        tools = await adapter.list_tools()
        logger.debug(f"工具：{[t.name for t in tools]}")
        agent = create_agent(model=llm, tools=tools, system_prompt="你是商家客服助手，只根据工具返回的内容回答，不要编造。问题里同时有订单号和退货、换货、政策时，必须先调 check_order 查订单，再调 search_shop_docs 查政策，两个结果都拿到再回答，缺一个都不行。")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "订单 20261116001 退了怎么办？"}]})
        for m in result["messages"]:
            logger.debug(f"{m.type}: {getattr(m, 'tool_calls', None) or m.content}")

asyncio.run(main())
