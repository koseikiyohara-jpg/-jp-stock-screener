import io
import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE = "https://api.jquants.com/v2"

st.set_page_config(page_title="日本株スクリーナー", page_icon="📈", layout="wide")

class JQuants:
    def __init__(self, api_key: str):
        self.headers = {"x-api-key": api_key.strip()}

    def get(self, path: str, params: dict | None = None) -> list[dict[str, Any]]:
        rows = []
        params = dict(params or {})
        while True:
            r = requests.get(BASE + path, headers=self.headers, params=params, timeout=35)
            if r.status_code == 401:
                raise RuntimeError("J-Quants APIキーを確認してください。")
            if r.status_code == 403:
                raise RuntimeError("このデータは現在のJ-Quants契約プランでは利用できません。")
            if r.status_code == 429:
                raise RuntimeError("J-Quants APIのレート制限に達しました。")
            r.raise_for_status()
            js = r.json()
            rows.extend(js.get("data", []))
            key = js.get("pagination_key")
            if not key:
                break
            params["pagination_key"] = key
        return rows


def code4(x):
    s = re.sub(r"[^0-9]", "", str(x).replace(".0", ""))
    return s[:4] if len(s) >= 4 else s.zfill(4)


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def latest_fy(rows):
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    if "DiscDate" in df:
        df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    if "CurPerType" in df:
        fy = df[df["CurPerType"].astype(str).eq("FY")]
        if not fy.empty:
            df = fy
    sort_cols = [c for c in ["DiscDate", "DiscNo"] if c in df]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.iloc[-1].to_dict()


def parse_override(data):
    emp, rev, debt = {}, {}, {}
    if not data:
        return emp, rev, debt
    raw = data.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="cp932")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "code" not in df:
        return emp, rev, debt
    df["code"] = df["code"].map(code4)
    if "employees" in df:
        emp = dict(zip(df["code"], pd.to_numeric(df["employees"], errors="coerce")))
    if "revenue" in df:
        rev = dict(zip(df["code"], pd.to_numeric(df["revenue"], errors="coerce")))
    if "interest_bearing_debt" in df:
        debt = dict(zip(df["code"], pd.to_numeric(df["interest_bearing_debt"], errors="coerce")))
    return emp, rev, debt


def event_stats(prices, event_dates):
    if prices.empty:
        return {}
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date")
    close_col = "AdjC" if "AdjC" in prices and prices["AdjC"].notna().any() else "C"
    prices[close_col] = pd.to_numeric(prices[close_col], errors="coerce")
    vals = []
    for ed in sorted(set(pd.to_datetime(event_dates, errors="coerce").dropna())):
        b = prices[prices["Date"] <= ed]
        a = prices[prices["Date"] > ed]
        if b.empty or a.empty:
            continue
        p0, p1 = b.iloc[-1][close_col], a.iloc[0][close_col]
        if np.isfinite(p0) and np.isfinite(p1) and p0 != 0:
            vals.append((p1 / p0 - 1) * 100)
    if not vals:
        return {}
    s = pd.Series(vals, dtype=float)
    return {
        "決算回数": len(s),
        "翌日上昇率(%)": round((s > 0).mean() * 100, 1),
        "平均騰落率(%)": round(s.mean(), 2),
        "中央値(%)": round(s.median(), 2),
        "最大上昇(%)": round(s.max(), 2),
        "最大下落(%)": round(s.min(), 2),
    }


def demo_data():
    return pd.DataFrame([
        {"コード":"7203","会社名":"デモA","売上高(億円)":480000,"従業員数":380000,"1人当たり売上(万円)":1263,"有利子負債(億円)":310000,"決算回数":8,"翌日上昇率(%)":62.5,"平均騰落率(%)":1.8,"最大下落(%)":-4.2},
        {"コード":"6758","会社名":"デモB","売上高(億円)":130000,"従業員数":110000,"1人当たり売上(万円)":1182,"有利子負債(億円)":25000,"決算回数":8,"翌日上昇率(%)":75.0,"平均騰落率(%)":2.4,"最大下落(%)":-3.1},
        {"コード":"9984","会社名":"デモC","売上高(億円)":70000,"従業員数":65000,"1人当たり売上(万円)":1077,"有利子負債(億円)":170000,"決算回数":8,"翌日上昇率(%)":50.0,"平均騰落率(%)":0.6,"最大下落(%)":-8.7},
    ])

st.title("📈 日本株スクリーナー")
st.caption("従業員1人当たり売上高・有利子負債・決算発表翌日の株価反応をまとめて比較")

with st.sidebar:
    st.header("データ設定")
    mode = st.radio("モード", ["デモ", "J-Quants実データ"])
    api_key = ""
    if mode == "J-Quants実データ":
        api_key = st.text_input("J-Quants APIキー", type="password")
    codes_text = st.text_area("証券コード", "7203, 6758, 9984")
    years = st.slider("決算統計の期間（年）", 1, 10, 5)
    override = st.file_uploader("従業員数などの補完CSV（任意）", type=["csv"])
    run = st.button("スクリーニング実行", type="primary", use_container_width=True)

if not run:
    st.info("左側で条件を設定して「スクリーニング実行」を押してください。まずはデモで動作確認できます。")
    st.stop()

if mode == "デモ":
    result = demo_data()
else:
    if not api_key.strip():
        st.error("J-Quants APIキーを入力してください。")
        st.stop()
    codes = []
    for raw in re.split(r"[,\s、]+", codes_text):
        if raw.strip():
            c = code4(raw)
            if c not in codes:
                codes.append(c)
    if not codes:
        st.error("証券コードを入力してください。")
        st.stop()
    if len(codes) > 40:
        st.error("一度に40銘柄までです。")
        st.stop()

    emp_map, rev_map, debt_map = parse_override(override)
    jq = JQuants(api_key)
    rows = []
    progress = st.progress(0)
    status = st.empty()
    for i, code in enumerate(codes):
        status.write(f"{code} を取得中…")
        item = {"コード": code}
        try:
            sums = jq.get("/fins/summary", {"code": code})
            latest = latest_fy(sums)
            revenue = num(latest.get("Sales"))
            if code in rev_map and np.isfinite(rev_map[code]):
                revenue = float(rev_map[code])
            employees = emp_map.get(code, np.nan)
            debt = debt_map.get(code, np.nan)

            cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
            if sums:
                sdf = pd.DataFrame(sums)
                dcol = "DiscDate" if "DiscDate" in sdf else ("SchDate" if "SchDate" in sdf else None)
                dates = pd.to_datetime(sdf[dcol], errors="coerce").dropna().tolist() if dcol else []
                dates = [d for d in dates if d >= cutoff]
            else:
                dates = []
            if dates:
                start = (min(dates) - timedelta(days=10)).strftime("%Y%m%d")
                end = (pd.Timestamp.today() + timedelta(days=2)).strftime("%Y%m%d")
                prices = pd.DataFrame(jq.get("/equities/bars/daily", {"code": code, "from": start, "to": end}))
            else:
                prices = pd.DataFrame()
            stats = event_stats(prices, dates)

            per_emp = revenue / employees if np.isfinite(revenue) and np.isfinite(employees) and employees > 0 else np.nan
            item.update({
                "会社名": latest.get("CoName") or latest.get("CompanyName") or "",
                "売上高(億円)": round(revenue / 1e8, 1) if np.isfinite(revenue) else np.nan,
                "従業員数": int(employees) if np.isfinite(employees) else np.nan,
                "1人当たり売上(万円)": round(per_emp / 1e4, 1) if np.isfinite(per_emp) else np.nan,
                "有利子負債(億円)": round(debt / 1e8, 1) if np.isfinite(debt) else np.nan,
                **stats,
            })
        except Exception as e:
            item["エラー"] = str(e)
        rows.append(item)
        progress.progress((i + 1) / len(codes))
    status.empty()
    result = pd.DataFrame(rows)

st.subheader("スクリーニング結果")

c1, c2, c3 = st.columns(3)
with c1:
    min_sales_emp = st.number_input("1人当たり売上 最低（万円）", min_value=0.0, value=0.0, step=100.0)
with c2:
    max_debt = st.number_input("有利子負債 上限（億円・0=無制限）", min_value=0.0, value=0.0, step=100.0)
with c3:
    min_win = st.number_input("決算翌日 上昇率 最低（%）", min_value=0.0, max_value=100.0, value=0.0, step=5.0)

filtered = result.copy()
if "1人当たり売上(万円)" in filtered and min_sales_emp > 0:
    filtered = filtered[pd.to_numeric(filtered["1人当たり売上(万円)"], errors="coerce") >= min_sales_emp]
if "有利子負債(億円)" in filtered and max_debt > 0:
    filtered = filtered[pd.to_numeric(filtered["有利子負債(億円)"], errors="coerce") <= max_debt]
if "翌日上昇率(%)" in filtered and min_win > 0:
    filtered = filtered[pd.to_numeric(filtered["翌日上昇率(%)"], errors="coerce") >= min_win]

st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button("CSVをダウンロード", filtered.to_csv(index=False).encode("utf-8-sig"), "stock_screening.csv", "text/csv")

st.caption("従業員数・有利子負債はJ-Quantsの契約プランや開示データの都合で取得できない場合があります。その場合はCSV補完を利用してください。")
