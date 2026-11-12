# mini-react-agent

手写 ReAct 循环：模型调工具、结果塞回消息列表、没工具调用就停，最多 5 轮。

工具：`get_current_time`（查时间）、`calculate`（算数）、`convert_currency`（美元按汇率换人民币）。

## 运行

```bash
pip install langchain-openai python-dotenv loguru
python mini_agent.py
```

需要 `.env` 里配好 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL_NAME`（不要提交 `.env`）。

## 例子

问：「现在几点？再帮我算 128 美元按 7.2 换成人民币是多少」
输出：现在是 2026年11月12日 15:42:37，128 美元 × 7.2 = 921.6 元人民币。
