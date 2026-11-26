from loguru import logger
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# u1、u2 各存一条「上次那单」
store.put(("u1", "memories"), "last_order", {"order_id": "20261116001", "text": "订单 20261116001 已发货，正在运输中"})
store.put(("u2", "memories"), "last_order", {"order_id": "20261116002", "text": "订单 20261116002 已签收"})
store.put(("u1", "preferences"), "pref-1", {"text": "用户喜欢红色连衣裙"})

# 验证 u1 搜不到 u2 的记忆
u1_items = store.search(("u1", "memories"))
logger.debug(f"u1 的记忆：{[(it.key, it.value['order_id']) for it in u1_items]}")
u1_ids = [it.value["order_id"] for it in u1_items]
logger.debug(f"u1 能看到 20261116002 吗：{'20261116002' in u1_ids}")

# 前缀 ("u1",) 列出 u1 全部记忆
all_u1 = store.search(("u1",))
logger.debug(f"前缀 ('u1',) 列出：{[(it.key, list(it.namespace)) for it in all_u1]}")
