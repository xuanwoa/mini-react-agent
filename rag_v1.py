import os

import httpx
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from loguru import logger

# Jina 直连不通，走本机 127.0.0.1:7890 代理
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

# 第一步：切块 + 向量化
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents([Document(page_content=d) for d in shop_docs])
logger.debug(f"切完一共 {len(chunks)} 块")

# check_embedding_ctx_length=False：让 LangChain 直接发原文给 Jina，
# 不要先转成 OpenAI 的 token id，否则 Jina 报 422
embeddings = OpenAIEmbeddings(model=os.getenv("JINA_MODEL_NAME"),
                              base_url=os.getenv("JINA_BASE_URL"),
                              api_key=os.getenv("JINA_API_KEY"),
                              http_client=http_client,
                              check_embedding_ctx_length=False)
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# 第二步：检索 + 回答
def ask(question):
    docs = retriever.invoke(question)
    for d in docs:
        logger.debug(f"搜到：{d.page_content}")
    context = "\n".join(d.page_content for d in docs)
    answer = llm.invoke(f"只根据以下资料回答，不要编造：\n{context}\n\n问题：{question}")
    logger.debug(answer.content)


ask("买了 5 天能退货吗？")
ask("退了怎么办？")
