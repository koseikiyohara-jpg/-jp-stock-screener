
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="日本株スクリーナー",
    page_icon="📈",
    layout="wide",
)

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
JQ_BASE = "https://api.jquants.com/v2"


# =========================
# 共通処理
# =========================

def code4(value: Any) -> str:
    s = re.sub(r"[^0-9]", "", str(value).replace(".0", ""))
    if len(s) >= 4:
        return s[:4]
    return s.zfill(4)


def parse_codes(text: str) -> list[str]:
    codes: list[str] = []
    for part in re.split(r"[,\s、]+", text):
        if not part.strip():
            continue
        c = code4(part)
        if c and c not in codes:
            codes.append(c)
    return codes


def num(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(str(value).replace(",", ""))
    except Exception:
        return np.nan


def event_stats(close: pd.Series, event_dates: list[pd.Timestamp]) -> dict[str, Any]:
    if close is None or close.empty:
        return {}

    s = pd.to_numeric(close, errors="coerce").dropna()
    if s.empty:
        return {}

    idx = pd.to_datetime(s.index, errors="coerce")
    s.index = idx
    s = s[~s.index.isna()].sort_index()

    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)

    vals: list[float] = []

    for raw in event_dates:
        try:
            d = pd.Timestamp(raw)
            if d.tzinfo is not None:
                d = d.tz_localize(None)
            d = d.normalize()
        except Exception:
            continue

        before = s[s.index.normalize() <= d]
        after = s[s.index.normalize() > d]

        if before.empty or after.empty:
            continue

        p0 = float(before.iloc[-1])
        p1 = float(after.iloc[0])

        if p0 == 0:
            continue

        vals.append((p1 / p0 - 1) * 100)

    if not vals:
        return {}

    x = pd.Series(vals, dtype=float)

    return {
        "決算回数": int(len(x)),
        "翌日上昇率(%)": round(float((x > 0).mean() * 100), 1),
        "平均騰落率(%)": round(float(x.mean()), 2),
        "中央値(%)": round(float(x.median()), 2),
        "最大上昇(%)": round(float(x.max()), 2),
        "最大下落(%)": round(float(x.min()), 2),
    }


def show_result_filters(df: pd.DataFrame, prefix: str) -> None:
    st.subheader("スクリーニング結果")

    c1, c2, c3 = st.columns(3)

    with c1:
        min_sales_emp = st.number_input(
            "1人当たり売上 最低（万円）",
            min_value=0.0,
            value=0.0,
            step=100.0,
            key=f"{prefix}_min_sales_emp",
        )

    with c2:
        max_debt = st.number_input(
            "有利子負債 上限（億円・0=無制限）",
            min_value=0.0,
            value=0.0,
            step=100.0,
            key=f"{prefix}_max_debt",
        )

    with c3:
        min_win_rate = st.number_input(
            "決算翌日 上昇率 最低（%）",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            key=f"{prefix}_min_win_rate",
        )

    out = df.copy()

    if "1人当たり売上(万円)" in out.columns and min_sales_emp > 0:
        v = pd.to_numeric(out["1人当たり売上(万円)"], errors="coerce")
        out = out[v >= min_sales_emp]

    if "有利子負債(億円)" in out.columns and max_debt > 0:
        v = pd.to_numeric(out["有利子負債(億円)"], errors="coerce")
        out = out[v <= max_debt]

    if "翌日上昇率(%)" in out.columns and min_win_rate > 0:
        v = pd.to_numeric(out["翌日上昇率(%)"], errors="coerce")
        out = out[v >= min_win_rate]

    st.dataframe(out, use_container_width=True, hide_index=True)

    st.download_button(
        "CSVをダウンロード",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{prefix}_screening.csv",
        mime="text/csv",
        key=f"{prefix}_download",
    )


# =========================
# 無料データ版
# yfinance経由
# =========================

def first_available_row(df: pd.DataFrame, row_names: list[str]) -> float:
    if df is None or df.empty:
        return np.nan

    for name in row_names:
        if name in df.index:
            vals = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0])

    return np.nan


def yf_earnings_dates(ticker: yf.Ticker, years: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []

    try:
        ed = ticker.get_earnings_dates(limit=max(12, years * 8))
        if ed is not None and not ed.empty:
            for idx in ed.index:
                try:
                    d = pd.Timestamp(idx)
                    if d.tzinfo is not None:
                        d = d.tz_localize(None)
                    dates.append(d)
                except Exception:
                    pass
    except Exception:
        pass

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    return [d for d in dates if d >= cutoff]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_free_one(code: str, years: int) -> dict[str, Any]:
    symbol = f"{code}.T"
    ticker = yf.Ticker(symbol)

    info: dict[str, Any] = {}
    try:
        info = ticker.info or {}
    except Exception:
        pass

    company = info.get("longName") or info.get("shortName") or ""

    employees = num(info.get("fullTimeEmployees"))
    revenue = num(info.get("totalRevenue"))
    debt = num(info.get("totalDebt"))

    if not np.isfinite(revenue):
        try:
            revenue = first_available_row(
                ticker.financials,
                ["Total Revenue", "Operating Revenue"],
            )
        except Exception:
            pass

    if not np.isfinite(debt):
        try:
            debt = first_available_row(
                ticker.balance_sheet,
                [
                    "Total Debt",
                    "Long Term Debt And Capital Lease Obligation",
                    "Long Term Debt",
                    "Current Debt",
                ],
            )
        except Exception:
            pass

    dates = yf_earnings_dates(ticker, years)
    stats: dict[str, Any] = {}

    if dates:
        try:
            start = (min(dates) - timedelta(days=15)).strftime("%Y-%m-%d")
            end = (pd.Timestamp.today() + timedelta(days=2)).strftime("%Y-%m-%d")
            hist = ticker.history(start=start, end=end, auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist:
                stats = event_stats(hist["Close"], dates)
        except Exception:
            pass

    per_emp = (
        revenue / employees
        if np.isfinite(revenue) and np.isfinite(employees) and employees > 0
        else np.nan
    )

    return {
        "コード": code,
        "会社名": company,
        "売上高(億円)": round(revenue / 1e8, 1) if np.isfinite(revenue) else np.nan,
        "従業員数": int(employees) if np.isfinite(employees) else np.nan,
        "1人当たり売上(万円)": round(per_emp / 1e4, 1) if np.isfinite(per_emp) else np.nan,
        "有利子負債(億円)": round(debt / 1e8, 1) if np.isfinite(debt) else np.nan,
        **stats,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_free_many(codes: tuple[str, ...], years: int) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []

    for code in codes:
        try:
            rows.append(fetch_free_one(code, years))
        except Exception as e:
            rows.append({"コード": code, "エラー": str(e)})

    fetched_at = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M")
    return pd.DataFrame(rows), fetched_at


# =========================
# J-Quants版
# =========================

class JQuants:
    def __init__(self, api_key: str):
        self.headers = {"x-api-key": api_key.strip()}

    def get(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        params = dict(params or {})

        while True:
            r = requests.get(
                JQ_BASE + path,
                headers=self.headers,
                params=params,
                timeout=35,
            )

            if r.status_code == 400:
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text[:500]
                raise RuntimeError(f"400 Bad Request: {detail}")

            if r.status_code == 401:
                raise RuntimeError("401: APIキーを確認してください。")

            if r.status_code == 403:
                raise RuntimeError("403: 現在の契約プランでは利用できないデータです。")

            if r.status_code == 429:
                raise RuntimeError("429: APIレート制限に達しました。少し待って再実行してください。")

            r.raise_for_status()

            js = r.json()
            rows.extend(js.get("data", []))

            pagination_key = js.get("pagination_key")
            if not pagination_key:
                break

            params["pagination_key"] = pagination_key

        return rows


def latest_fy(rows: list[dict[str, Any]]) -> dict[str, Any]:
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

    sort_cols = [c for c in ["DiscDate", "DiscNo"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.iloc[-1].to_dict()


def get_saved_jq_key() -> str:
    try:
        return str(st.secrets.get("JQUANTS_API_KEY", "")).strip()
    except Exception:
        return ""


def jq_window(plan: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.today().normalize()

    if plan == "Free":
        latest = today - pd.Timedelta(weeks=12)
        oldest = latest - pd.DateOffset(years=2)
    elif plan == "Light":
        latest = today
        oldest = today - pd.DateOffset(years=5)
    elif plan == "Standard":
        latest = today
        oldest = today - pd.DateOffset(years=10)
    else:
        latest = today
        oldest = pd.Timestamp("2000-01-01")

    return oldest, latest


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_jq_many(
    api_key: str,
    codes: tuple[str, ...],
    years: int,
    plan: str,
) -> tuple[pd.DataFrame, str]:

    jq = JQuants(api_key)
    oldest_allowed, latest_allowed = jq_window(plan)
    rows: list[dict[str, Any]] = []

    for code in codes:
        item: dict[str, Any] = {"コード": code}

        try:
            summaries = jq.get("/fins/summary", {"code": code})
            latest = latest_fy(summaries)

            revenue = num(latest.get("Sales"))
            employees = np.nan
            debt = np.nan

            cutoff = max(
                pd.Timestamp.today().normalize() - pd.DateOffset(years=years),
                oldest_allowed,
            )

            dates: list[pd.Timestamp] = []

            if summaries:
                sdf = pd.DataFrame(summaries)
                dcol = None

                if "DiscDate" in sdf.columns:
                    dcol = "DiscDate"
                elif "SchDate" in sdf.columns:
                    dcol = "SchDate"

                if dcol:
                    tmp = pd.to_datetime(sdf[dcol], errors="coerce").dropna()
                    dates = [
                        d for d in tmp
                        if cutoff <= d <= latest_allowed
                    ]

            stats: dict[str, Any] = {}

            if dates:
                start_date = max(min(dates) - timedelta(days=10), oldest_allowed)
                end_date = latest_allowed

                prices = pd.DataFrame(
                    jq.get(
                        "/equities/bars/daily",
                        {
                            "code": code,
                            "from": start_date.strftime("%Y%m%d"),
                            "to": end_date.strftime("%Y%m%d"),
                        },
                    )
                )

                if not prices.empty:
                    date_col = "Date" if "Date" in prices.columns else None
                    close_col = next(
                        (
                            c
                            for c in ["AdjC", "C", "Close", "AdjustmentClose"]
                            if c in prices.columns
                        ),
                        None,
                    )

                    if date_col and close_col:
                        prices[date_col] = pd.to_datetime(
                            prices[date_col],
                            errors="coerce",
                        )
                        prices = prices.dropna(subset=[date_col]).sort_values(date_col)

                        close_series = pd.Series(
                            pd.to_numeric(prices[close_col], errors="coerce").values,
                            index=prices[date_col],
                        )

                        stats = event_stats(close_series, dates)

            per_emp = (
                revenue / employees
                if np.isfinite(revenue) and np.isfinite(employees) and employees > 0
                else np.nan
            )

            item.update(
                {
                    "会社名": latest.get("CoName") or latest.get("CompanyName") or "",
                    "売上高(億円)": round(revenue / 1e8, 1) if np.isfinite(revenue) else np.nan,
                    "従業員数": employees,
                    "1人当たり売上(万円)": round(per_emp / 1e4, 1) if np.isfinite(per_emp) else np.nan,
                    "有利子負債(億円)": debt,
                    **stats,
                }
            )

        except Exception as e:
            item["エラー"] = str(e)

        rows.append(item)

    fetched_at = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M")
    return pd.DataFrame(rows), fetched_at


# =========================
# ページ
# =========================

def home_page() -> None:
    st.title("📈 日本株スクリーナー")
    st.write("無料版・J-Quants版・比較検証を1つのアプリに分けました。")

    st.success("最新版：無料版 / J-Quants版 / 比較・検証 の3ページ構成")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🆓 無料データ版")
        st.write("APIキー不要。検証・バックアップ用途。")
        st.caption("yfinance経由で取得するため、欠損や仕様変更の影響を受ける場合があります。")

    with c2:
        st.subheader("🏦 J-Quants版")
        st.write("本番確認用。契約プランを選択できます。")
        st.caption("Streamlit SecretsにAPIキーを保存すれば毎回入力不要です。")

    with c3:
        st.subheader("🔎 比較・検証")
        st.write("同じ銘柄を無料版とJ-Quants版で横並び比較。")
        st.caption("将来J-Quantsを解約する場合の移行判断にも使えます。")


def free_page() -> None:
    st.title("🆓 無料データ版")
    st.caption("検証・バックアップ用。J-Quants契約なしで動きます。")

    with st.sidebar:
        st.header("無料版 設定")
        codes_text = st.text_area(
            "証券コード",
            "7203, 6758, 9984",
            key="free_codes",
        )
        years = st.slider(
            "決算統計の期間（年）",
            1,
            5,
            2,
            key="free_years",
        )

        if st.button("無料版キャッシュを更新", use_container_width=True):
            fetch_free_many.clear()
            fetch_free_one.clear()
            st.success("無料版キャッシュを削除しました。")

        run = st.button(
            "無料版スクリーニング実行",
            type="primary",
            use_container_width=True,
        )

    if not run:
        st.info("左側で条件を設定して実行してください。")
        return

    codes = parse_codes(codes_text)

    if not codes:
        st.error("証券コードを入力してください。")
        return

    with st.spinner("無料データを取得中…"):
        result, fetched_at = fetch_free_many(tuple(codes), years)

    st.caption(
        f"最終取得: {fetched_at} "
        "（同じ条件の結果は最大7日間再利用）"
    )

    show_result_filters(result, "free")

    st.warning(
        "無料版は検証用です。従業員数・有利子負債・決算日などが"
        "取得できない銘柄があります。"
    )


def jquants_page() -> None:
    st.title("🏦 J-Quants版")
    st.caption("J-Quants APIを使う本番ページ")

    saved_key = get_saved_jq_key()

    with st.sidebar:
        st.header("J-Quants 設定")

        if saved_key:
            api_key = saved_key
            st.success("APIキー保存済み")
        else:
            api_key = st.text_input(
                "J-Quants APIキー",
                type="password",
                key="jq_api_key",
            )
            st.caption("Secretsに保存すると毎回入力不要になります。")

        plan = st.selectbox(
            "契約プラン",
            ["Free", "Light", "Standard", "Premium"],
            index=1,
            key="jq_plan",
        )

        codes_text = st.text_area(
            "証券コード",
            "7203, 6758, 9984",
            key="jq_codes",
        )

        years = st.slider(
            "決算統計の期間（年）",
            1,
            10,
            5,
            key="jq_years",
        )

        if st.button("J-Quantsキャッシュを更新", use_container_width=True):
            fetch_jq_many.clear()
            st.success("J-Quantsキャッシュを削除しました。")

        run = st.button(
            "J-Quantsスクリーニング実行",
            type="primary",
            use_container_width=True,
        )

    if not run:
        st.info("左側で条件を設定して実行してください。")
        return

    if not api_key:
        st.error("J-Quants APIキーを入力してください。")
        return

    codes = parse_codes(codes_text)

    with st.spinner("J-Quantsデータを取得中…"):
        result, fetched_at = fetch_jq_many(
            api_key,
            tuple(codes),
            years,
            plan,
        )

    st.caption(
        f"最終取得: {fetched_at} "
        "（同じ条件の結果は最大7日間再利用）"
    )

    show_result_filters(result, "jq")


def compare_page() -> None:
    st.title("🔎 比較・検証")
    st.caption("無料版とJ-Quants版を同じ銘柄で比較します。")

    saved_key = get_saved_jq_key()

    with st.sidebar:
        st.header("比較 設定")

        code = code4(
            st.text_input(
                "証券コード",
                "7203",
                key="compare_code",
            )
        )

        years = st.slider(
            "決算統計の期間（年）",
            1,
            5,
            2,
            key="compare_years",
        )

        plan = st.selectbox(
            "J-Quants契約プラン",
            ["Free", "Light", "Standard", "Premium"],
            index=1,
            key="compare_plan",
        )

        if saved_key:
            api_key = saved_key
            st.success("J-Quants APIキー保存済み")
        else:
            api_key = st.text_input(
                "J-Quants APIキー",
                type="password",
                key="compare_api_key",
            )

        run = st.button(
            "比較実行",
            type="primary",
            use_container_width=True,
        )

    if not run:
        st.info("左側で1銘柄を指定して比較してください。")
        return

    with st.spinner("無料版を取得中…"):
        free_df, free_at = fetch_free_many((code,), years)

    jq_df = pd.DataFrame()
    jq_at = "-"

    if api_key:
        with st.spinner("J-Quants版を取得中…"):
            jq_df, jq_at = fetch_jq_many(
                api_key,
                (code,),
                years,
                plan,
            )

    if free_df.empty:
        st.error("無料版の取得に失敗しました。")
        return

    free_row = free_df.iloc[0].to_dict()
    jq_row = jq_df.iloc[0].to_dict() if not jq_df.empty else {}

    fields = [
        "会社名",
        "売上高(億円)",
        "従業員数",
        "1人当たり売上(万円)",
        "有利子負債(億円)",
        "決算回数",
        "翌日上昇率(%)",
        "平均騰落率(%)",
        "最大上昇(%)",
        "最大下落(%)",
    ]

    compare_rows: list[dict[str, Any]] = []

    for field in fields:
        free_value = free_row.get(field, np.nan)
        jq_value = jq_row.get(field, np.nan)

        difference = np.nan

        if field != "会社名":
            try:
                if pd.notna(free_value) and pd.notna(jq_value):
                    difference = round(float(free_value) - float(jq_value), 2)
            except Exception:
                pass

        compare_rows.append(
            {
                "項目": field,
                "無料版": free_value,
                "J-Quants版": jq_value,
                "差（無料-JQ）": difference,
            }
        )

    st.dataframe(
        pd.DataFrame(compare_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"無料版取得: {free_at} / J-Quants取得: {jq_at}"
    )

    if not api_key:
        st.warning(
            "J-Quants APIキー未入力のため、無料版のみ取得しています。"
        )


# =========================
# ナビゲーション
# =========================

pages = {
    "メニュー": [
        st.Page(home_page, title="ホーム", icon="🏠", default=True),
    ],
    "スクリーナー": [
        st.Page(free_page, title="無料データ版", icon="🆓"),
        st.Page(jquants_page, title="J-Quants版", icon="🏦"),
        st.Page(compare_page, title="比較・検証", icon="🔎"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
