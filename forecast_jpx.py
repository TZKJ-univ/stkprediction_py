#!/usr/bin/env python3
import argparse, sys, time
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

import pandas as pd, numpy as np, yfinance as yf, lightgbm as lgb, json
import json
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

FEAT_COLS: list[str] = []   # features order will be stored after first build

CODES_FILE = Path("jpx_codes.txt")
DATA_DIR   = Path("feather"); DATA_DIR.mkdir(exist_ok=True)
PERIOD, INTERVAL, CHUNK = "10y", "1d", 200
LAGS, SHIFT = [1,5,22,66], 22          # 1か月 = 22営業日
TRAIN_SPLIT_YEARS = 0.5   # 最後の0.5年（6ヶ月）を検証用に使う

SECTOR_FILE = Path("sectors.json")
SECTOR_DICT: dict[str, str] = {}      # filled in main()

def load_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    Return {ticker: sector}.  Unknown → "Unknown".
    Cache to sectors.json so we do not hit Yahoo every run.
    """
    mp = {}
    if SECTOR_FILE.exists():
        try:
            mp = json.loads(SECTOR_FILE.read_text())
        except Exception:
            pass
    updated = False
    for t in tickers:
        if t not in mp:
            try:
                info = yf.Ticker(t).info
                mp[t] = info.get("sector", "Unknown") or "Unknown"
            except Exception:
                mp[t] = "Unknown"
            updated = True
    if updated:
        SECTOR_FILE.write_text(json.dumps(mp, ensure_ascii=False))
    return mp

SECTOR_FILE = Path("sectors.json")
FUND_FILE   = Path("fundamentals.json")

def load_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """
    Return {ticker: {"PER": ..., "PBR": ..., "DivYield": ...}}.
    Cache to fundamentals.json to avoid repeated calls.
    """
    mp = {}
    if FUND_FILE.exists():
        try:
            mp = json.loads(FUND_FILE.read_text())
        except Exception:
            pass
    updated = False
    for t in tickers:
        if t not in mp:
            try:
                info = yf.Ticker(t).info
                # capture all numeric fundamentals from yfinance info
                num_info = {k: v for k, v in info.items() if isinstance(v, (int, float))}
                mp[t] = num_info
            except Exception:
                mp[t] = {}
            updated = True
    if updated:
        FUND_FILE.write_text(json.dumps(mp, ensure_ascii=False))
    return mp


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
    # ensure index is datetime for train/test split
    mat.index = pd.to_datetime(mat.index)
    return mat

def features(mat: pd.DataFrame, funds: dict[str, dict]) -> pd.DataFrame:
    # 基本ラグ
    feats = {f"lag_{l}": mat.shift(l) for l in LAGS}

    # 移動平均・ボリンジャーバンド
    feats["sma_22"] = mat.rolling(22).mean()
    feats["sma_50"] = mat.rolling(50).mean()
    rolling_std     = mat.rolling(22).std()
    feats["bb_width"] = (rolling_std * 2) / feats["sma_22"]   # バンド幅比率

    # ボラティリティ(22d)
    feats["vol_22"] = mat.pct_change().rolling(22).std()
    # 価格と移動平均の比率
    feats["price_sma22_ratio"] = mat / feats["sma_22"]
    feats["price_sma50_ratio"] = mat / feats["sma_50"]

    # MACD (12,26)
    ema12 = mat.ewm(span=12, adjust=False).mean()
    ema26 = mat.ewm(span=26, adjust=False).mean()
    feats["macd"] = ema12 - ema26

    # RSI
    feats["rsi_14"] = 100 - 100 / (1 + mat.pct_change().rolling(14).mean())

    X = pd.concat(feats, axis=1).stack(future_stack=True).dropna().reset_index()
    # ensure Date column is datetime64 for merge
    X["Date"] = pd.to_datetime(X["Date"])
    X["dayofweek"] = pd.to_datetime(X["Date"]).dt.dayofweek.astype("int8")
    X["month"]     = pd.to_datetime(X["Date"]).dt.month.astype("int8")

    # --- 業種をカテゴリとして追加 ---
    global SECTOR_DICT
    X["Sector"] = X["Ticker"].map(SECTOR_DICT).fillna("Unknown")

    # --- map static fundamentals ---
    # map all numeric fundamentals dynamically
    if funds:
        # get list of fundamental keys from the first ticker
        fund_keys = list(next(iter(funds.values())).keys())
        for key in fund_keys:
            X[key] = X["Ticker"].map(lambda t, k=key: funds.get(t, {}).get(k))
            # fill missing fundamentals with 0
            X[key].fillna(0, inplace=True)

    y = (
        (np.log(mat.shift(-SHIFT)) - np.log(mat))
        .stack()
        .reindex(X.set_index(["Date", "Ticker"]).index)
    )
    X["target"] = y.values
    # ±100% (= 約 ±0.693 in log) 以上の外れ値を除外
    X = X[X["target"].abs() <= 1.0]
    # keep rows even if fundamentals have missing (filled with 0), only drop if target missing
    return X[X["target"].notna()]

def _prep_xy(df: pd.DataFrame, cat: list[str]):
    """return X(num) for xgb, X(cat) for catboost, y, cat_idx, w"""
    global FEAT_COLS
    X = df.drop(["Date", "target"], axis=1)
    if not FEAT_COLS:
        FEAT_COLS = X.columns.tolist()
    X = X[FEAT_COLS]          # 固定順
    y = df["target"].values
    # --- xgb 用: cat→int ---
    X_num = X.copy()
    for c in cat:
        X_num[c] = X_num[c].cat.codes
    # --- catboost 用: cat→str ---
    X_cat = X.copy()
    for c in cat:
        X_cat[c] = X_cat[c].astype(str)
    cat_idx = [X.columns.get_loc(c) for c in cat]
    w = df["weight"].values if "weight" in df else None
    return X, X_num, X_cat, y, cat_idx, w

def train_models(df: pd.DataFrame, num_rounds: int, cat: list[str], params_lgb: dict):
    """return (lgb_model, xgb_model, cat_model)"""
    X, X_num, X_cat, y, cat_idx, w = _prep_xy(df, cat)

    # --- LightGBM ---
    dtrain_lgb = lgb.Dataset(X, label=y, weight=w, categorical_feature=cat, free_raw_data=False)
    mdl_lgb = lgb.train(params_lgb, dtrain_lgb, num_boost_round=num_rounds)

    # --- XGBoost ---
    dtrain_xgb = xgb.DMatrix(X_num, label=y, weight=w)
    params_xgb = dict(
        objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=max(1, cpu_count()-1),
        tree_method="hist"
    )
    mdl_xgb = xgb.train(params_xgb, dtrain_xgb, num_rounds)

    # --- CatBoost ---
    mdl_cat = CatBoostRegressor(
        iterations=num_rounds,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        verbose=False,
        random_seed=42
    )
    mdl_cat.fit(X_cat, y, sample_weight=w, cat_features=cat_idx)

    return mdl_lgb, mdl_xgb, mdl_cat

# ---- Optuna で LightGBM ハイパラ探索 ----
def tune_lgbm_params(df_train: pd.DataFrame, cat: list[str], n_trials: int = 60):
    X = df_train.drop(["Date", "target"], axis=1)
    y = df_train["target"].values
    w = df_train["weight"].values
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = dict(
            objective="regression",
            metric="l2",
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 4, 14),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50),
            num_threads=max(1, cpu_count()-1),
            verbosity=-1,
        )
        scores, iters = [], []
        for tr_idx, vl_idx in tscv.split(X):
            dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], weight=w[tr_idx],
                              categorical_feature=cat, free_raw_data=False)
            dvl = lgb.Dataset(X.iloc[vl_idx], label=y[vl_idx], weight=w[vl_idx],
                              categorical_feature=cat, free_raw_data=False)
            m = lgb.train(params, dtr, 500,
                          valid_sets=[dvl],
                          valid_names=["valid"],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            # compute validation predictions at best iteration and evaluate R2
            preds = m.predict(X.iloc[vl_idx], num_iteration=m.best_iteration)
            r2 = r2_score(y[vl_idx], preds)
            scores.append(r2)
            iters.append(m.best_iteration)
        trial.set_user_attr("best_iters", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update(objective="regression", metric=["r2"], num_threads=max(1, cpu_count()-1), verbosity=-1)
    best_iter  = study.best_trial.user_attrs["best_iters"]
    return best_params, best_iter

def predict(mat: pd.DataFrame, models: tuple, weights: np.ndarray) -> pd.DataFrame:
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

    global SECTOR_DICT
    X["Sector"] = X["Ticker"].map(SECTOR_DICT).fillna("Unknown").astype("category")

    X_lgb = X.copy()
    X_xgb = X.copy()
    # --- align columns with training feature order ---
    global FEAT_COLS
    missing = [c for c in FEAT_COLS if c not in X_lgb.columns]
    for m in missing:
        X_lgb[m] = 0
        X_xgb[m] = 0
    X_lgb = X_lgb[FEAT_COLS]
    X_xgb = X_xgb[FEAT_COLS].copy()
    for c in ["Ticker", "Sector"]:
        X_xgb[c] = X_xgb[c].cat.codes

    lgb_model, xgb_model, cat_model = models
    p_lgb = lgb_model.predict(X_lgb)
    p_xgb = xgb_model.predict(xgb.DMatrix(X_xgb))
    p_cat = cat_model.predict(X_lgb)
    preds = weights[0]*p_lgb + weights[1]*p_xgb + weights[2]*p_cat

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
    global SECTOR_DICT
    SECTOR_DICT = load_sector_map(cds)

    FUNDAMENTALS = load_fundamentals(cds)

    for i in tqdm(range(0, len(cds), CHUNK), desc="download"):
        update_chunk(cds[i:i + CHUNK], i // CHUNK)

    mat = price_matrix(cds)

    # --- 現在値が 100 円未満の銘柄を除外 ---
    current_prices = mat.iloc[-1]
    mat = mat.loc[:, current_prices >= 100]
    if mat.empty:
        sys.exit("No tickers >= 100円")

    df = features(mat, FUNDAMENTALS)

    # ---- サンプルウェイト (0.99 ** day_diff) ----
    max_dt = pd.to_datetime(df["Date"]).max()
    df["weight"] = 0.995 ** ((max_dt - pd.to_datetime(df["Date"])).dt.days)

    # ---- 時系列 hold‑out ----
    months = int(TRAIN_SPLIT_YEARS * 12)
    # 特徴量作成後のデータに基づいてホールドアウトを計算
    max_feat_date = pd.to_datetime(df["Date"]).max()
    split_date = max_feat_date - pd.DateOffset(months=months)
    df_train = df[pd.to_datetime(df["Date"]) <= split_date].copy()
    df_test  = df[pd.to_datetime(df["Date"]) >  split_date].copy()
    if df_test.empty:
        print("テスト用データが存在しないはずがありません（特徴量データ基準で分割）。")
        df_test = df_train.copy()

    cat = ["Ticker", "Sector"]
    for c in cat:
        df_train[c] = df_train[c].astype("category")

    for c in cat:
        for df_ in (df_train, df_test):
            df_[c] = pd.Categorical(df_[c], categories=df_train[c].cat.categories)

    # ---- Optuna で LightGBM ハイパラ探索 (30 trials) ----
    best_params, best_iter = tune_lgbm_params(df_train, cat, n_trials=60)
    num_boost = max(50, best_iter)
    models = train_models(df_train, num_boost, cat, best_params)
    # ---- Calculate ensemble weights by Test MSE inverse ----
    def _model_preds(df_part):
        Xp = df_part.drop(["Date", "target"], axis=1)
        p_lgb = models[0].predict(Xp)

        Xp_num = Xp.copy()
        for c in cat:
            Xp_num[c] = Xp_num[c].cat.codes
        p_xgb = models[1].predict(xgb.DMatrix(Xp_num))

        # prepare categorical input for CatBoost: fill NaNs and convert to string
        Xp_cat = Xp.copy()
        for c in cat:
            # ensure "Unknown" category exists before filling
            if "Unknown" not in Xp_cat[c].cat.categories:
                Xp_cat[c] = Xp_cat[c].cat.add_categories("Unknown")
            Xp_cat[c] = Xp_cat[c].fillna("Unknown").astype(str)
        p_cat = models[2].predict(Xp_cat)
        return np.vstack([p_lgb, p_xgb, p_cat])

    preds_stack = _model_preds(df_test)
    mses = ((preds_stack - df_test["target"].values)**2).mean(axis=1)
    weights = 1 / mses
    weights /= weights.sum()
    cat_levels = {c: df_train[c].cat.categories for c in cat}

    # ---- 指標 ----
    def _metric(df_part, name):
        df_part = df_part.copy()
        for c in cat:
            df_part[c] = pd.Categorical(df_part[c], categories=cat_levels[c])
        y_true = df_part["target"].values
        y_pred = models[0].predict(df_part.drop(["Date", "target"], axis=1))
        mse  = mean_squared_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return f"{name}  MSE: {mse:.6f} | R2: {r2:.4f}"

    print(_metric(df_train, "Train"), flush=True)
    print(_metric(df_test,  "Test "), flush=True)
    top = predict(mat, models, weights).head(20)
    top.to_csv(args.csv,index=False)
    print(top.to_string(index=False, formatters={
        "Current": "{:.2f}".format,
        "Predicted": "{:.2f}".format,
        "Ratio": "{:.2%}".format,
    }))

if __name__=="__main__":
    main()