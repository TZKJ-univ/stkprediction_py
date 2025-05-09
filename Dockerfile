# ベースイメージに公式の Python スリム版を使用
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Tokyo

# システム依存ライブラリのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3-pip python3-setuptools \
      build-essential \
      git && \
    ln -s /usr/bin/python3.12 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを /app に設定
WORKDIR /app

# 依存関係をコピーしてインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードを全てコピー
COPY . .

# デフォルトの実行コマンド
ENTRYPOINT ["python", "forecast_jpx.py"]