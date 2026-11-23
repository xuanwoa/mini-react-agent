import os

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger
from pydantic import BaseModel, Field

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

# 店铺资料（agent_v4 那 10 条，照搬）
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
    return ORDERS.get(order_id, "查询不到订单，请确认订单号？")


class RouteDecision(BaseModel):
    needs_order: bool = Field(description="问题里有没有订单号、要不要查订单状态")
    needs_policy: bool = Field(description="问题里有没有问退货、换货、运费、保修这类政策")
    order_id: str | None = Field(default=None, description="问题里的 11 位订单号，没有就填 None")


classifier = llm.with_structured_output(RouteDecision)

agent = create_agent(model=llm, tools=[search_shop_docs, check_order],
                     middleware=[SummarizationMiddleware(model=llm, trigger=("messages", 6), keep=("messages", 2))],
                     checkpointer=InMemorySaver(), )

config = {"configurable": {"thread_id": "eval-1"}}


def ask_agent(question: str) -> str:
    """调 agent_v4 的流程：先分类，再按分类提示去答。"""
    route = classifier.invoke(f"判断下面这个问题需要查什么：{question}")
    hint = "这个问题只查店铺资料。"  # ★ 先给默认值，分支再覆盖它
    if route.needs_order and route.needs_policy:
        hint = f"这个问题要同时查订单 {route.order_id} 和退货政策，两个工具都要调用"
    elif route.needs_order:
        hint = f"这个问题只查订单 {route.order_id}"
    result = agent.invoke({"messages": [{"role": "user", "content": f"{question}\n(提示:{hint})"}]}, config=config)
    return result["messages"][-1].content


# 裁判：结构化打分
class JudgeScore(BaseModel):
    score: int = Field(description="答到的要点数")
    total: int = Field(description="期望要点总数")
    missed: list[str] = Field(description="漏掉的期望要点，没漏就填空列表")
    retrieval_wrong: bool = Field(description="回答里引用的资料是不是错的、答非所问")


judge = llm.with_structured_output(JudgeScore)

# 20 条评估题：问题 + 期望要点（照抄 eval_questions.md）
EVAL_SET = [
    ("买了 5 天能退货吗？", ["7 天内无理由退货", "运费买家承担"]),
    ("退货要寄到哪里？", ["杭州西湖区仓储中心"]),
    ("退货运费谁出？", ["买家承担"]),
    ("换货怎么换？", ["7 天内可换货", "流程与退货相同"]),
    ("超过 7 天还能退吗？", ["超过 7 天不能无理由退货", "质量问题走售后"]),
    ("订单 20261116001 到哪了？", ["已发货", "运输中", "预计明天送达"]),
    ("订单 20261116002 签收了吗？", ["已签收", "2026-11-15 14:30"]),
    ("我的订单发货了吗？", ["先问用户要订单号", "回答里没有编造的 11 位订单号"]),
    ("订单 20261116001 退了怎么办？", ["订单状态", "退货政策"]),
    ("订单 20261116002 能退吗？", ["已签收", "7 天退货政策"]),
    ("订单 20261116001 能换货吗？", ["已发货", "运输中", "7 天内可换货", "流程与退货相同"]),
    ("订单 20261116002 换货怎么换？", ["已签收", "7 天内可换货", "流程与退货相同"]),
    ("订单 20261116001 退货运费谁出？", ["已发货", "运输中", "退货运费买家承担"]),
    ("买了 10 天发现有质量问题能退吗？", ["超过 7 天不能无理由退货", "质量问题走售后"]),
    ("订单号是 20261116999，到哪了？", ["查不到该订单", "请确认订单号"]),
    ("我的订单退货要寄到哪里？", ["杭州西湖区仓储中心"]),
    ("保修多久？人为损坏保修吗？", ["保修 1 年", "人为损坏不保修"]),
    ("能开发票吗？", ["可开电子发票", "下单时备注即可"]),
    ("今天几点前下单能当天发货？", ["16:00 前下单当天发货", "16:00 后次日发货"]),
    ("客服几点在线？", ["每天 9:00-21:00 在线"]),
]

logger.debug("题号 | 得分 | 漏了哪条要点 | 检索搜错没有")
total_score = 0
total_points = 0

for i, (question, points) in enumerate(EVAL_SET, start=1):
    answer = ask_agent(question)
    verdict = judge.invoke(
        f"你是评分员。对照期望要点给下面的回答打分。\n"
        f"问题：{question}\n"
        f"期望要点：{points}\n"
        f"助手回答：{answer}\n"
        f"每答到一个要点得 1 分；漏掉的要点列出来；如果回答引用的资料明显不对、答非所问，retrieval_wrong 填 true。"
    )
    total_score += verdict.score
    total_points += verdict.total
    logger.debug(
        f"第 {i} 题 | {verdict.score}/{verdict.total} | 漏了：{verdict.missed} | 检索错：{verdict.retrieval_wrong}")
    logger.debug(f"  回答：{answer[:120]}")

logger.debug(f"总分：{total_score}/{total_points}")
