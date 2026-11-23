import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from loguru import logger

load_dotenv(Path(__file__).resolve().parent / ".env")

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),
)


async def main():
    server_path = Path(__file__).resolve().parent / "mcp_server.py"
    async with MCPAdapter(server_path) as adapter:
        tools = await adapter.list_tools()
        logger.debug(f"拿到 {len(tools)} 个 MCP 工具：{[t.name for t in tools]}")
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="你是商家客服助手，只根据工具返回的内容回答，不要编造。",
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "订单 20261116001 退了怎么办？"}]}
        )
        for m in result["messages"]:
            calls = getattr(m, "tool_calls", None)
            if calls:
                logger.debug(f"[{m.type}] 调了工具：{[c['name'] for c in calls]}")
            elif m.content:
                logger.debug(f"[{m.type}] {m.content}")
        logger.debug(f"最终回答：{result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
