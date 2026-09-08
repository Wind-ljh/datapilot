# DataPilot API + 演示界面（构建时安装全部依赖，首次启动自动生成演示库）
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE ./
COPY server ./server
COPY eval ./eval
COPY training ./training
COPY app ./app
COPY data/mini_eval.jsonl data/fewshot_examples.jsonl ./data/
COPY scripts ./scripts

RUN pip install --no-cache-dir .

# 首次启动自动生成演示库（已存在则跳过）
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
