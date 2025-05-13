# CUDA 12.6.3 + Ubuntu24.04 をベースにする
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Tokyo
ENV PATH="/opt/venv/bin:$PATH"

# システム依存ライブラリと venv 用ツールのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/venv

WORKDIR /app

# 依存を先にコピーしてキャッシュを有効活用
COPY requirements.txt ./

# 仮想環境内でインストール
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードを全コピー
COPY . .

# 実行コマンド
ENTRYPOINT ["python", "forecast_jpx.py"]
CMD ["--csv", "result.csv"]
