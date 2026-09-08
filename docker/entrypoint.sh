#!/bin/sh
set -e
if [ ! -f /app/data/ecommerce.duckdb ]; then
  echo "[entrypoint] 生成演示库 data/ecommerce.duckdb ..."
  python -m server.core.seed_data --rows 100000 --path /app/data/ecommerce.duckdb
fi
exec "$@"
