# core.py
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# ============================================================
# KONSTANTA BROKER
# ============================================================
BANDAR_FOREIGN  = {"ak", "bk", "zp", "yu", "tp", "bq", "xa"}
BANDAR_DOMESTIK = {"mg", "rf", "hp", "if", "sq", "ni"}

BROKER_LIST = [
    "XL", "CC", "AK", "ZP", "YP", "YU", "PD", "BK", "MG", "XC", "CP", "LG",
    "SQ", "NI", "KZ", "AZ", "DH", "RX", "OD", "KK", "KI", "BB", "IF", "GR",
    "BQ", "DR", "HD", "TP", "YB", "SS", "EP", "XA", "AI", "DX", "RB", "HP",
    "AG", "BS", "YJ", "AP", "AO", "DP", "BR", "RF", "CD", "AT", "IN", "FZ",
    "II", "SH", "PO", "PG", "PC", "MU", "LS", "IU", "PP", "FS", "TS", "IH",
    "ID", "MI", "RO", "ES", "EL", "YO", "SA", "GA", "ZR", "AF", "OK", "SF",
    "RG", "AR", "AN", "DU", "PI", "PF", "RS", "QA", "GI", "JB", "BF", "IC",
    "TF", "AH", "IT", "AD", "FO", "DD", "GW",
]

# ============================================================
# ANALISIS BANDAR
# ============================================================
def weighted_average_price(data):
    total_lot = sum(lot for _, lot, _ in data)
    if total_lot == 0:
        return 0
    total_value = sum(lot * price for _, lot, price in data)
    return total_value / total_lot


def find_consistent_brokers(data_periods):
    if not data_periods:
        return set()
    sets = [set(broker for broker, _, _ in period) for period in data_periods]
    return set.intersection(*sets) if sets else set()


def classify(brokers):
    foreign  = [b for b in brokers if b.lower() in BANDAR_FOREIGN]
    domestik = [b for b in brokers if b.lower() in BANDAR_DOMESTIK]
    other    = [b for b in brokers
                if b.lower() not in BANDAR_FOREIGN
                and b.lower() not in BANDAR_DOMESTIK]
    return foreign, domestik, other


def analyze_broker_summary(data_periods):
    """
    data_periods: list of dict {buyers: [(broker, lot, price), ...],
                                 sellers: [(broker, lot, price), ...]}
    """
    buyers_all  = [p["buyers"]  for p in data_periods]
    sellers_all = [p["sellers"] for p in data_periods]

    consistent_buyers  = find_consistent_brokers(buyers_all)
    consistent_sellers = find_consistent_brokers(sellers_all)

    foreign_buyers,  dom_buyers,  other_buyers  = classify(consistent_buyers)
    foreign_sellers, dom_sellers, other_sellers = classify(consistent_sellers)

    has_bandar_buy  = bool(foreign_buyers  or dom_buyers)
    has_bandar_sell = bool(foreign_sellers or dom_sellers)

    def format_bandar_list(foreign, domestic):
        parts = []
        if foreign:
            parts.append(f"Foreign: {', '.join(foreign)}")
        if domestic:
            parts.append(f"Domestik: {', '.join(domestic)}")
        return "; ".join(parts)

    if has_bandar_buy and has_bandar_sell:
        status = "KONFLIK BANDAR (AKUMULASI & DISTRIBUSI)"
        buy_text  = format_bandar_list(foreign_buyers,  dom_buyers)
        sell_text = format_bandar_list(foreign_sellers, dom_sellers)
        bandar_text = f"Buy: {buy_text} | Sell: {sell_text}"
        risk   = "HIGH (adu bandar / perang harga)"
        korban = "Pasar tidak menentu, terjadi tarik-menarik antar bandar"

    elif has_bandar_buy:
        status = "AKUMULASI"
        bandar_text = format_bandar_list(foreign_buyers, dom_buyers)
        risk   = "HIGH (aksi bandar akumulasi)"
        korban = "Top seller tidak konsisten (ritel atau bandar lain)"

    elif has_bandar_sell:
        status = "DISTRIBUSI"
        bandar_text = format_bandar_list(foreign_sellers, dom_sellers)
        risk   = "HIGH (aksi bandar distribusi)"
        korban = "Top buyer tidak konsisten (ritel atau bandar lain)"

    elif other_buyers:
        status = "AKUMULASI LOW RISK"
        bandar_text = f"Non-bandar (buy): {', '.join(other_buyers)}"
        risk   = "LOW (bukan bandar utama)"
        korban = "-"

    elif other_sellers:
        status = "DISTRIBUSI LOW RISK"
        bandar_text = f"Non-bandar (sell): {', '.join(other_sellers)}"
        risk   = "LOW (bukan bandar utama)"
        korban = "-"

    else:
        status = "TIDAK JELAS / SIDEWAYS"
        bandar_text = "-"
        risk   = "-"
        korban = "-"

    avg_buy  = weighted_average_price(buyers_all[-1])  if buyers_all[-1]  else 0
    avg_sell = weighted_average_price(sellers_all[-1]) if sellers_all[-1] else 0

    return {
        "status": status,
        "bandar_text": bandar_text,
        "risk": risk,
        "korban": korban,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
    }


# ============================================================
# TEKNIKAL (yFinance)
# ============================================================
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_historical_data(symbol, period="6mo"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return None, "⚠️ Gagal mengambil data. Periksa kode saham."
        return hist, None
    except Exception as e:
        return None, f"⚠️ Error: {e}"


def get_technical_data(hist):
    if hist is None or hist.empty:
        return None, "Data kosong"

    closes  = hist["Close"]
    volumes = hist["Volume"]

    if len(closes) < 50:
        return None, "⚠️ Data kurang dari 50 hari."

    ma5  = closes.rolling(window=5).mean()
    ma10 = closes.rolling(window=10).mean()
    ma20 = closes.rolling(window=20).mean()
    ma50 = closes.rolling(window=50).mean()
    rsi  = calculate_rsi(closes)

    avg20 = volumes.rolling(window=20).mean().iloc[-1]
    return {
        "ma5":  ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma50": ma50,
        "rsi":  rsi,
        "current_price": closes.iloc[-1],
        "volume_today":  int(volumes.iloc[-1]) if not np.isnan(volumes.iloc[-1]) else 0,
        "avg_volume_20": int(avg20) if not np.isnan(avg20) else 0,
        "hist": hist,
    }, None


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================
def find_support_resistance(hist, order=5):
    high = hist["High"].values
    low  = hist["Low"].values

    swing_high_idx = argrelextrema(high, np.greater_equal, order=order)[0]
    swing_low_idx  = argrelextrema(low,  np.less_equal,    order=order)[0]

    high_levels = high[swing_high_idx] if len(swing_high_idx) > 0 else np.array([])
    low_levels  = low[swing_low_idx]   if len(swing_low_idx)  > 0 else np.array([])

    current_price = hist["Close"].iloc[-1]
    resistance_levels = sorted([h for h in high_levels if h > current_price])[:3] if len(high_levels) else []
    support_levels    = sorted([l for l in low_levels  if l < current_price], reverse=True)[:3] if len(low_levels) else []

    return {
        "resistance": resistance_levels if resistance_levels else None,
        "support":    support_levels    if support_levels    else None,
        "swing_high": high[swing_high_idx[-1]] if len(swing_high_idx) > 0 else None,
        "swing_low":  low[swing_low_idx[-1]]   if len(swing_low_idx)  > 0 else None,
    }


# ============================================================
# ANALISIS TEKNIKAL
# ============================================================
def analyze_trend(current_price, ma5, ma10, ma20, ma50):
    if current_price > ma5 > ma10 > ma20 > ma50:
        return "UPTREND KUAT (harga > MA5 > MA10 > MA20 > MA50)"
    elif current_price > ma5 and ma5 > ma10 and current_price > ma20 and current_price > ma50:
        return "UPTREND (harga di atas MA, MA5 > MA10)"
    elif current_price > ma5 and current_price > ma10 and current_price > ma20 and current_price > ma50:
        return "UPTREND LEMAH (harga di atas semua MA, tapi urutan MA belum sempurna)"
    elif current_price < ma5 < ma10 < ma20 < ma50:
        return "DOWNTREND KUAT (harga < MA5 < MA10 < MA20 < MA50)"
    elif current_price < ma5 and current_price < ma10 and current_price < ma20 and current_price < ma50:
        return "DOWNTREND LEMAH"
    else:
        return "SIDEWAYS / KONSOLIDASI"


def analyze_volume(today_volume, avg_volume_20):
    if avg_volume_20 == 0:
        return "Volume rata-rata 20 hari tidak valid."
    ratio = today_volume / avg_volume_20
    if ratio > 2:
        return (f"Volume hari ini = {today_volume:,}, rata-rata 20 hari = {avg_volume_20:,} "
                f"→ Rasio {ratio:.2f}x. Volume sangat tinggi. Breakout atau aksi bandar.")
    elif ratio > 1.2:
        return (f"Volume hari ini = {today_volume:,}, rata-rata 20 hari = {avg_volume_20:,} "
                f"→ Rasio {ratio:.2f}x. Volume di atas rata-rata. Minat meningkat.")
    elif ratio < 0.5:
        return (f"Volume hari ini = {today_volume:,}, rata-rata 20 hari = {avg_volume_20:,} "
                f"→ Rasio {ratio:.2f}x. Volume rendah. Sideways atau kurang minat.")
    else:
        return (f"Volume hari ini = {today_volume:,}, rata-rata 20 hari = {avg_volume_20:,} "
                f"→ Rasio {ratio:.2f}x. Volume normal.")


def analyze_rsi(rsi):
    if rsi > 70:
        return f"RSI {rsi:.1f} → Overbought. Potensi koreksi."
    elif rsi < 30:
        return f"RSI {rsi:.1f} → Oversold. Potensi rebound."
    else:
        return f"RSI {rsi:.1f} → Netral."