#!/usr/bin/env python3
import argparse, sys, time
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

import pandas as pd, numpy as np, yfinance as yf, lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

CODES_FILE = Path("jpx_codes.txt")
DATA_DIR   = Path("feather"); DATA_DIR.mkdir(exist_ok=True)
PERIOD, INTERVAL, CHUNK = "10y", "1d", 200
LAGS, SHIFT = [1,5,22,66], 22          # 1か月 = 22営業日
TRAIN_SPLIT_YEARS = 1   # 最後の 1 年を検証用に使う

def codes() -> list[str]:
    """
    Read jpx_codes.txt and return list like ["7203.T", "6758.T", ...]
    Only keep lines that are 4–5 digit numbers.
    """
    if not CODES_FILE.exists():
        sys.exit("jpx_codes.txt not found")
    out = []
    for line in CODES_FILE.read_text().splitlines():
        code = line.strip()
        if code.isdigit() and 4 <= len(code) <= 5:
            out.append(f"{code.zfill(4)}.T")
    if not out:
        sys.exit("No valid codes")
    return out

def update_ticker_csv(ticker: str, max_retry: int = 5):
    # DEPRECATED:
    """
    Keep <DATA_DIR>/<ticker>.csv with historical Close prices.
    Retry with exponential back‑off when yfinance hits rate‑limit (429/401).
    """
    fcsv = DATA_DIR / f"{ticker}.csv"
    start = "2015-01-01"
    if fcsv.exists():
        last = pd.read_csv(fcsv, usecols=["Date"]).Date.max()
        start = (pd.to_datetime(last) + pd.Timedelta(days=1)).date()

    for r in range(max_retry):
        try:
            if fcsv.exists():
                df = yf.download(
                    ticker,
                    start=start,
                    interval=INTERVAL,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            else:
                # 初回は期間指定で広めに取得
                df = yf.download(
                    ticker,
                    period=PERIOD,
                    interval=INTERVAL,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            if not df.empty:
                break  # success
        except Exception:
            pass  # fall through to retry
        wait = 2 ** r
        time.sleep(wait)
    else:
        return  # all retries failed or empty

    # --- Close 列を Series として取得 ---
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(1)):
            s = df.xs("Close", level=1, axis=1)
        elif ("Adj Close" in df.columns.get_level_values(1)):
            s = df.xs("Adj Close", level=1, axis=1)
        else:
            return
        series = s.squeeze().dropna()
    else:
        col = "Close" if "Close" in df else "Adj Close"
        series = df[col].dropna()

    if series.empty:
        return

    # -- Create tidy DataFrame ---
    if isinstance(series, pd.Series):
        price = series.to_frame(name="Close").reset_index()
    else:  # just in case series returned as DataFrame with single column
        price = series.reset_index().rename(columns={series.columns[0]: "Close"})

    # Ensure date column name
    if price.columns[0] != "Date":
        price.rename(columns={price.columns[0]: "Date"}, inplace=True)

    price.insert(1, "Ticker", ticker)
    price["Date"] = pd.to_datetime(price["Date"]).dt.strftime("%Y-%m-%d")

    if fcsv.exists():
        (pd.concat([pd.read_csv(fcsv), price])
         .drop_duplicates(subset=["Date"])
         .to_csv(fcsv, index=False))
    else:
        price.to_csv(fcsv, index=False)

def update_chunk(chunk: list[str], idx: int):
    f = DATA_DIR / f"{idx}.feather"

    start = "2015-01-01"
    if f.exists():
        last = pd.read_feather(f, columns=["Date"]).Date.max()
        start = (pd.to_datetime(last) + pd.Timedelta(days=1)).date()
        today = datetime.now().date()
        if start > today:
            # up to date; nothing new to fetch
            return

    df = yf.download(
        " ".join(chunk),
        start=start,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if df.empty:
        return

    # ----- Close が無いときは Adj Close を使う -----
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        if "Close" in lvl1:
            closes = df.xs("Close", level=1, axis=1)
        elif "Adj Close" in lvl1:
            closes = df.xs("Adj Close", level=1, axis=1)
        else:
            return
        closes = (
            closes.stack()
            .reset_index()
            .rename(columns={0: "Close"})
        )
    else:
        closes = df[["Close"]] if "Close" in df else pd.DataFrame()
        if closes.empty and "Adj Close" in df:
            closes = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
        if closes.empty:
            return
        closes = (
            closes.reset_index()
            .rename(columns={"index": "Date"})
            .assign(Ticker=chunk[0])
        )

    closes.columns = ["Date", "Ticker", "Close"]

    # ----------- 保存 -----------
    if f.exists():
        (pd.concat([pd.read_feather(f), closes])
           .drop_duplicates(subset=["Date", "Ticker"])
           .to_feather(f))
    else:
        closes.to_feather(f)

def price_matrix(cds: list[str]) -> pd.DataFrame:
    """Build Close matrix from feather/<idx>.feather files created by update_chunk."""
    files = list(DATA_DIR.glob("*.feather"))
    if not files:
        sys.exit("no feather data")
    df = pd.concat(pd.read_feather(f) for f in files)
    mat = (
        df.pivot(index="Date", columns="Ticker", values="Close")
          .sort_index()
          .ffill()
    )
    return mat

def features(mat: pd.DataFrame) -> pd.DataFrame:
    # 基本ラグ
    feats = {f"lag_{l}": mat.shift(l) for l in LAGS}

    # 移動平均・ボリンジャーバンド
    feats["sma_22"] = mat.rolling(22).mean()
    feats["sma_50"] = mat.rolling(50).mean()
    rolling_std     = mat.rolling(22).std()
    feats["bb_width"] = (rolling_std * 2) / feats["sma_22"]   # バンド幅比率

    # MACD (12,26)
    ema12 = mat.ewm(span=12, adjust=False).mean()
    ema26 = mat.ewm(span=26, adjust=False).mean()
    feats["macd"] = ema12 - ema26

    # RSI
    feats["rsi_14"] = 100 - 100 / (1 + mat.pct_change().rolling(14).mean())

    X = pd.concat(feats, axis=1).stack(future_stack=True).dropna().reset_index()
    X["dayofweek"] = pd.to_datetime(X["Date"]).dt.dayofweek.astype("int8")
    X["month"]     = pd.to_datetime(X["Date"]).dt.month.astype("int8")

    y = (
        (np.log(mat.shift(-SHIFT)) - np.log(mat))
        .stack()
        .reindex(X.set_index(["Date", "Ticker"]).index)
    )
    X["target"] = y.values
    # ±100% (= 約 ±0.693 in log) 以上の外れ値を除外
    X = X[X["target"].abs() <= 1.0]
    return X.dropna()

def train(df: pd.DataFrame, num_rounds: int):
    cat = ["Ticker"]
    for c in cat: df[c] = df[c].astype("category")
    dtrain = lgb.Dataset(df.drop(["Date","target"],axis=1), label=df.target,
                         categorical_feature=cat, free_raw_data=False)
    params = dict(objective="regression", learning_rate=0.05,
                  num_leaves=63, metric="mae",
                  num_threads=cpu_count()-1, verbosity=-1)
    return lgb.train(params, dtrain, num_boost_round=num_rounds)

def predict(mat: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
    last_feat = {}

    for l in LAGS:
        last_feat[f"lag_{l}"] = mat.shift(l).tail(1)

    last_feat["sma_22"] = mat.rolling(22).mean().tail(1)
    last_feat["sma_50"] = mat.rolling(50).mean().tail(1)
    rolling_std = mat.rolling(22).std().tail(1)
    last_feat["bb_width"] = (rolling_std * 2) / last_feat["sma_22"]

    ema12 = mat.ewm(span=12, adjust=False).mean().tail(1)
    ema26 = mat.ewm(span=26, adjust=False).mean().tail(1)
    last_feat["macd"] = ema12 - ema26

    last_feat["rsi_14"] = 100 - 100 / (1 + mat.pct_change().rolling(14).mean()).tail(1)

    X = (
        pd.concat(last_feat, axis=1)
        .stack(future_stack=True)   # suppress FutureWarning
        .reset_index()
        .drop(["Date"], axis=1)
    )
    X["dayofweek"] = datetime.now().weekday()
    X["month"] = datetime.now().month
    X["Ticker"] = X["Ticker"].astype("category")

    preds = model.predict(X)

    current_prices = mat.iloc[-1]
    curr = X["Ticker"].map(current_prices)

    # preds は log-return ⇒ 終値を復元
    pred_price = curr.values * np.exp(preds)
    ratio = np.exp(preds) - 1

    df_out = pd.DataFrame({
        "Ticker": X["Ticker"],
        "Current": curr.values,
        "Predicted": pred_price,
        "Ratio": ratio,
    })
    df_out = df_out[df_out["Ratio"] > 0]
    return df_out.sort_values("Ratio", ascending=False)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv",default="result.csv")
    args = p.parse_args()

    cds = codes()
    for i in tqdm(range(0, len(cds), CHUNK), desc="download"):
        update_chunk(cds[i:i + CHUNK], i // CHUNK)

    mat = price_matrix(cds)

    # --- 現在値が 100 円未満の銘柄を除外 ---
    current_prices = mat.iloc[-1]
    mat = mat.loc[:, current_prices >= 100]
    if mat.empty:
        sys.exit("No tickers >= 100円")

    df = features(mat)

    # ---- 時系列 hold‑out ----
    split_date = mat.index.max() - pd.DateOffset(years=TRAIN_SPLIT_YEARS)
    df_train = df[pd.to_datetime(df["Date"]) <= split_date].copy()
    df_test  = df[pd.to_datetime(df["Date"]) >  split_date].copy()

    # ---- LightGBM + walk‑forward CV ----
    cat = ["Ticker"]
    for c in cat:
        df_train[c] = df_train[c].astype("category")

    X_all = df_train.drop(["Date", "target"], axis=1)
    y_all = df_train["target"]

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    params = dict(objective="regression",
                  learning_rate=0.05,
                  num_leaves=63,
                  metric="mae",
                  num_threads=cpu_count() - 1,
                  verbosity=-1)
    best_iters = []
    for tr_idx, vl_idx in tscv.split(X_all):
        dtr = lgb.Dataset(X_all.iloc[tr_idx], label=y_all.iloc[tr_idx],
                          categorical_feature=cat, free_raw_data=False)
        dvl = lgb.Dataset(X_all.iloc[vl_idx], label=y_all.iloc[vl_idx],
                          categorical_feature=cat, free_raw_data=False)
        m = lgb.train(
            params,
            dtr,
            num_boost_round=500,
            valid_sets=[dvl],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        best_iters.append(m.best_iteration)

    num_boost = max(50, int(np.mean(best_iters)))
    model = train(df_train, num_boost)
    cat_levels = df_train["Ticker"].cat.categories

    # ---- 指標 ----
    def _metric(df_part, name):
        df_part = df_part.copy()
        for c in ["Ticker"]:
            df_part[c] = pd.Categorical(df_part[c], categories=cat_levels)
        y_true = df_part["target"].values
        y_pred = model.predict(df_part.drop(["Date", "target"], axis=1))
        mse  = mean_squared_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return f"{name}  MSE: {mse:.6f} | R2: {r2:.4f}"

    print(_metric(df_train, "Train"), flush=True)
    print(_metric(df_test,  "Test "), flush=True)
    top = predict(mat, model).head(20)
    top.to_csv(args.csv,index=False)
    print(top.to_string(index=False, formatters={
        "Current": "{:.2f}".format,
        "Predicted": "{:.2f}".format,
        "Ratio": "{:.2%}".format,
    }))

if __name__=="__main__":
    main()