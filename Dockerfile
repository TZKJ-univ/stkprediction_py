# ベースに NGC の公式 PyTorch コンテナを使う（CUDA 12.6 + cuDNN 最適化済み）
FROM nvcr.io/nvidia/pytorch:24.08-py3

# 必要ならタイムゾーンなど環境変数を設定
ENV TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ユーザーの Python 依存関係
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 実行
ENTRYPOINT ["python", "forecast_jpx.py"]
CMD ["--csv", "result.csv"]
