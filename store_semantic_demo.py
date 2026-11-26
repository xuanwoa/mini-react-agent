import os
import httpx
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

load_dotenv(Path(__file__).resolve().parent / ".env")
http_client = httpx.Client(proxy="http://127.0.0.1:7890", timeout=60.0)
embeddings = OpenAIEmbeddings(
    model=os.getenv("JINA_MODEL_NAME"),
    base_url=os.getenv("JINA_BASE_URL"),
    api_key=os.getenv("JINA_API_KEY"),
    http_client=http_client,
    check_embedding_ctx_length=False,
)

# dims 不用猜：先嵌一句话量出来
dims = len(embeddings.embed_query("test"))
logger.debug(f"Jina 维度：{dims}")

store = InMemoryStore(index={
    "embed": embeddings,
    "dims": dims,
    "fields": ["text"],
})

ns = ("u1", "memories")
store.put(ns, "order-1", {"text": "订单 20261116001 已发货，正在运输中", "order_id": "20261116001"})
store.put(ns, "order-2", {"text": "订单 20261116002 已签收", "order_id": "20261116002"})
store.put(ns, "pref-1", {"text": "用户喜欢红色连衣裙", "pref": "红色连衣裙"})

# 1. 不给 query：按存入顺序列出来
items = store.search(ns)
logger.debug(f"不给 query：{[it.key for it in items]}")

# 2. 给 query：按意思排，带分数
results = store.search(ns, query="我上次那单到哪了")
for r in results:
    logger.debug(f"score={r.score:.3f} key={r.key} value={r.value}")
