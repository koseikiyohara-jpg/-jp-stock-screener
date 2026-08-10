import io
import math
import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = "https://api.jquants.com/v2"

app = FastAPI(title="日本株スクリーナー", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


class JQuants:
    def __init__(self, api_key: str):
        self.headers = {"x-api-key": api_key.strip()}

    def get(self, path: str, params: dict | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        params = dict(params or {})
        while True:
            r = requests.get(BASE + path, headers=self.headers, params=params, timeout=35)
            if r.status_code == 401:
                raise HTTPException(401, "J-Quants APIキーを確認してください。")
            if r.status_code == 403:
                raise HTTPException(403, "このデータは現在のJ-Quants契約プランでは利用できません。")
            if r.status_code == 429:
                raise HTTPException(429, "J-Quants APIのレート制限に達しました。")
            r.raise_for_status()
            js = r.json()
            rows.extend(js.get("data", []))
            key = js.get("pagination_key")
            if not key:
                break
            params["pagination_key"] = key
        return rows


def code4(x: Any) -> str:
    s = str(x).strip().replace(".0", "")
    s = re.sub(r"[^0-9]", "", s)
    return s[:4] if len(s) >= 4 else s.zfill(4)


def to_num(x: Any) -> float:
    try:
        if x is None or x == "":
            return np.nan
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def clean_json_number(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    if hasattr(x, "isoformat") and not isinstance(x, str):
        try:
            return x.isoformat()
        except Exception:
            pass
    return x


def latest_fy_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    if "DiscDate" in df:
        df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    fy = df[df["CurPerType"].astype(str).eq("FY")] if "CurPerType" in df else df
    if fy.empty:
        fy = df
    sort_cols = [c for c in ["DiscDate", "DiscNo"] if c in fy]
    if sort_cols:
        fy = fy.sort_values(sort_cols)
    return fy.iloc[-1].to_dict()


DEBT_POS = [
    r"short.?term borrow", r"short.?term loan", r"current portion.*long.?term",
    r"long.?term borrow", r"long.?term loan", r"bonds payable", r"corporate bonds",
    r"commercial paper", r"interest.?bearing debt", r"borrowings", r"lease liabilit"
]
DEBT_NEG = [r"trade", r"receivable", r"asset", r"commitment", r"interest expense", r"cash flow"]


def infer_interest_bearing_debt(details_rows: list[dict]):
    if not details_rows:
        return np.nan, []
    df = pd.DataFrame(details_rows)
    if df.empty:
        return np.nan, []
    if "DiscDate" in df:
        df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    fy_mask = df.get("DocType", pd.Series("", index=df.index)).astype(str).str.contains("FY", case=False, na=False)
    cand = df[fy_mask] if fy_mask.any() else df
    sort_cols = [c for c in ["DiscDate", "DiscNo"] if c in cand]
    if sort_cols:
        cand = cand.sort_values(sort_cols)
    fs = cand.iloc[-1].get("FS", {}) or {}

    explicit, components = [], []
    for k, v in fs.items():
        key = str(k).lower()
        val = to_num(v)
        if not np.isfinite(val):
            continue
        if "interest-bearing debt" in key or "interest bearing debt" in key:
            explicit.append((k, val))
        elif any(re.search(p, key) for p in DEBT_POS) and not any(re.search(p, key) for p in DEBT_NEG):
            components.append((k, val))
    if explicit:
        return max(v for _, v in explicit), explicit

    seen, chosen = set(), []
    for k, v in components:
        rv = round(v, 2)
        if rv not in seen:
            seen.add(rv)
            chosen.append((k, v))
    return (float(sum(v for _, v in chosen)) if chosen else np.nan), chosen


def normalize_earnings_dates(rows: list[dict]) -> list[pd.Timestamp]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    col = "DiscDate" if "DiscDate" in df else ("SchDate" if "SchDate" in df else None)
    if not col:
        return []
    return pd.to_datetime(df[col], errors="coerce").dropna().tolist()


def event_stats(prices: pd.DataFrame, event_dates: list[pd.Timestamp]):
    if prices.empty:
        return {}, []
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date")
    close_col = "AdjC" if "AdjC" in prices and prices["AdjC"].notna().any() else "C"
    prices[close_col] = pd.to_numeric(prices[close_col], errors="coerce")

    out = []
    for ed in sorted(set(pd.to_datetime(event_dates, errors="coerce").dropna())):
        before_df = prices[prices["Date"] <= ed]
        after_df = prices[prices["Date"] > ed]
        if before_df.empty or after_df.empty:
            continue
        before, after = before_df.iloc[-1], after_df.iloc[0]
        p0, p1 = before[close_col], after[close_col]
        if not np.isfinite(p0) or not np.isfinite(p1) or p0 == 0:
            continue
        r = (p1 / p0 - 1) * 100
        out.append({
            "earnings_date": ed.strftime("%Y-%m-%d"),
            "base_date": before["Date"].strftime("%Y-%m-%d"),
            "next_trading_day": after["Date"].strftime("%Y-%m-%d"),
            "before_close": round(float(p0), 2),
            "next_close": round(float(p1), 2),
            "change_pct": round(float(r), 3),
        })
    if not out:
        return {}, out
    s = pd.Series([x["change_pct"] for x in out], dtype=float)
    stats = {
        "events": int(len(s)), "up_count": int((s > 0).sum()), "down_count": int((s < 0).sum()),
        "win_rate": float((s > 0).mean() * 100), "avg_change": float(s.mean()),
        "median_change": float(s.median()), "max_gain": float(s.max()), "max_loss": float(s.min()),
        "volatility": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }
    return stats, out


def parse_overrides(file_bytes: bytes | None):
    employees, revenue, debt = {}, {}, {}
    if not file_bytes:
        return employees, revenue, debt
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="cp932")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "code" not in df:
        return employees, revenue, debt
    df["code"] = df["code"].map(code4)
    if "employees" in df:
        employees = dict(zip(df["code"], pd.to_numeric(df["employees"], errors="coerce")))
    if "revenue" in df:
        revenue = dict(zip(df["code"], pd.to_numeric(df["revenue"], errors="coerce")))
    if "interest_bearing_debt" in df:
        debt = dict(zip(df["code"], pd.to_numeric(df["interest_bearing_debt"], errors="coerce")))
    return employees, revenue, debt


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/screen")
async def screen(
    api_key: str = Form(...),
    codes: str = Form(...),
    lookback_years: int = Form(5),
    overrides: UploadFile | None = File(None),
):
    if not api_key.strip():
        raise HTTPException(400, "APIキーを入力してください。")
    parsed_codes = []
    for raw in re.split(r"[,\s、]+", codes):
        if raw.strip():
            c = code4(raw)
            if c not in parsed_codes:
                parsed_codes.append(c)
    if not parsed_codes:
        raise HTTPException(400, "証券コードを1つ以上入力してください。")
    if len(parsed_codes) > 40:
        raise HTTPException(400, "一度に取得できるのは40銘柄までです。")
    lookback_years = max(1, min(10, int(lookback_years)))

    file_bytes = await overrides.read() if overrides else None
    emp_map, rev_override, debt_override = parse_overrides(file_bytes)
    jq = JQuants(api_key)
    results = []

    for code in parsed_codes:
        try:
            sums = jq.get("/fins/summary", {"code": code})
            latest = latest_fy_summary(sums)
            revenue = to_num(latest.get("Sales"))
            if code in rev_override and np.isfinite(rev_override[code]):
                revenue = float(rev_override[code])

            debt = np.nan
            debt_components = []
            debt_source = "not_available"
            if code in debt_override and np.isfinite(debt_override[code]):
                debt = float(debt_override[code]); debt_source = "csv"
                debt_components = [{"label": "CSV override", "amount": debt}]
            else:
                try:
                    details = jq.get("/fins/details", {"code": code})
                    debt, comps = infer_interest_bearing_debt(details)
                    if np.isfinite(debt): debt_source = "jquants"
                    debt_components = [{"label": str(k), "amount": float(v)} for k, v in comps]
                except Exception:
                    pass

            cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)
            event_dates = [d for d in normalize_earnings_dates(sums) if pd.Timestamp(d) >= cutoff]
            if event_dates:
                start = (min(event_dates) - timedelta(days=10)).strftime("%Y%m%d")
                end = (pd.Timestamp.today() + timedelta(days=2)).strftime("%Y%m%d")
                prices = pd.DataFrame(jq.get("/equities/bars/daily", {"code": code, "from": start, "to": end}))
            else:
                prices = pd.DataFrame()
            stats, events = event_stats(prices, event_dates)

            employees = emp_map.get(code, np.nan)
            sales_per_emp = revenue / employees if np.isfinite(revenue) and np.isfinite(employees) and employees > 0 else np.nan
            name = latest.get("CoName") or latest.get("CompanyName") or latest.get("CompanyNameEnglish") or ""
            item = {
                "code": code, "name": str(name or ""),
                "revenue_oku": revenue / 1e8 if np.isfinite(revenue) else None,
                "employees": int(employees) if np.isfinite(employees) else None,
                "sales_per_employee_man": sales_per_emp / 1e4 if np.isfinite(sales_per_emp) else None,
                "interest_bearing_debt_oku": debt / 1e8 if np.isfinite(debt) else None,
                "debt_source": debt_source, "debt_components": debt_components,
                "events_detail": events, **stats,
            }
            results.append({k: clean_json_number(v) for k, v in item.items()})
        except HTTPException:
            raise
        except Exception as e:
            results.append({"code": code, "name": "", "error": str(e), "events_detail": []})

    return JSONResponse({"results": results, "lookback_years": lookback_years})


@app.get("/api/demo")
def demo():
    return {
        "results": [
            {"code":"7203","name":"トヨタ自動車","revenue_oku":480367,"employees":383853,"sales_per_employee_man":1251.5,"interest_bearing_debt_oku":312000,"events":8,"up_count":5,"down_count":3,"win_rate":62.5,"avg_change":1.42,"median_change":0.88,"max_gain":7.91,"max_loss":-4.35,"volatility":3.78,"events_detail":[{"earnings_date":"2026-05-08","base_date":"2026-05-08","next_trading_day":"2026-05-11","before_close":2850,"next_close":2965,"change_pct":4.035}]},
            {"code":"6758","name":"ソニーグループ","revenue_oku":129571,"employees":112300,"sales_per_employee_man":1153.8,"interest_bearing_debt_oku":36500,"events":8,"up_count":6,"down_count":2,"win_rate":75.0,"avg_change":2.11,"median_change":1.74,"max_gain":8.26,"max_loss":-3.14,"volatility":3.31,"events_detail":[{"earnings_date":"2026-05-14","base_date":"2026-05-14","next_trading_day":"2026-05-15","before_close":3680,"next_close":3812,"change_pct":3.587}]},
            {"code":"7974","name":"任天堂","revenue_oku":18500,"employees":8120,"sales_per_employee_man":2278.3,"interest_bearing_debt_oku":820,"events":8,"up_count":4,"down_count":4,"win_rate":50.0,"avg_change":0.64,"median_change":0.31,"max_gain":9.02,"max_loss":-7.28,"volatility":5.14,"events_detail":[{"earnings_date":"2026-05-07","base_date":"2026-05-07","next_trading_day":"2026-05-08","before_close":12420,"next_close":12080,"change_pct":-2.738}]},
        ],
        "lookback_years": 5,
        "demo": True,
        "note": "表示値はUI確認用のサンプルです。投資判断には使用しないでください。"
    }
