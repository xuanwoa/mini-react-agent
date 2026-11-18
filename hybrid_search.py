import os

import httpx
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# 和 rag_v1.py 同一套配置：Jina 走本机 127.0.0.1:7890 代理
http_client = httpx.Client(proxy="http://127.0.0.1:7890", timeout=60.0)

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
    "超期退货：超过 7 天不再受理无理由退货，有质量问题走售后流程。",
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents([Document(page_content=d) for d in shop_docs])

embeddings = OpenAIEmbeddings(model=os.getenv("JINA_MODEL_NAME"),
                              base_url=os.getenv("JINA_BASE_URL"),
                              api_key=os.getenv("JINA_API_KEY"),
                              http_client=http_client,
                              check_embedding_ctx_length=False)
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)


# 路线一：向量检索——按意思找
def vector_search(question, k=3):
    results = vectorstore.similarity_search_with_score(question, k=k)
    logger.debug("向量路线：按意思找")
    for doc, score in results:
        logger.debug(f"  分数 {score:.4f}：{doc.page_content}")
    return [doc for doc, _ in results]


# 路线二：关键词检索——按原词找
# 中文没有空格分词，这里用最直白的办法：把问题切成 2 字词，
# 只要文档里出现这个词，就算命中一次，命中次数越多排越前。
def keyword_search(question, k=3):
    terms = [question[i:i + 2] for i in range(len(question) - 1)]
    scored = []
    for doc in chunks:
        hits = sum(1 for t in terms if t in doc.page_content)
        if hits > 0:
            scored.append((hits, doc))
    scored.sort(key=lambda x: -x[0])
    logger.debug("关键词路线：按原词找")
    for hits, doc in scored[:k]:
        logger.debug(f"  命中 {hits} 个词：{doc.page_content}")
    return [doc for _, doc in scored[:k]]


# 合并：RRF——两条路线各给一个名次分，加起来再排
# 第 1 名得 1/(60+1)，第 2 名得 1/(60+2)，以此类推，两边分数相加
def hybrid_search(question, k=3):
    vec_docs = vector_search(question)
    kw_docs = keyword_search(question)
    scores = {}
    for rank, doc in enumerate(vec_docs):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (60 + rank + 1)
    for rank, doc in enumerate(kw_docs):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (60 + rank + 1)
    merged = sorted(scores.items(), key=lambda x: -x[1])
    logger.debug("合并后 RRF：")
    for content, score in merged[:k]:
        logger.debug(f"  总分 {score:.4f}：{content}")
    return merged[:k]


logger.debug("========== 问题 1：退了怎么办？ ==========")
hybrid_search("退了怎么办？")

logger.debug("========== 问题 2：西湖区仓储中心电话多少？ ==========")
hybrid_search("西湖区仓储中心电话多少？")
