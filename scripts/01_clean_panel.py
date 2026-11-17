"""
Ghép panel 6 tháng từ dữ liệu Michigan Surveys of Consumers (SCA).

Input:  data/raw/sca_microdata_raw.csv  (342,345 quan sát x 44 biến, tải từ SDA archive)
Output: data/processed/panel_clean.csv  (mỗi dòng = 1 người có đủ 2 lần phỏng vấn cách nhau 6 tháng)

Biến ghép panel: bất kỳ dòng nào có IDPREV khác rỗng là một lần tái phỏng vấn;
IDPREV của dòng đó = ID của lần phỏng vấn ngay trước (6 tháng trước đó).
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/sca_microdata_raw.csv"
OUT_PATH = "data/processed/panel_clean.csv"

# Missing-data codes theo codebook (data/raw/sca_microdata_codebook.txt)
PX_MISSING = {-97, 96, 98, 99}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    # Các cột số bị đọc thành string do khoảng trắng đệm trong file gốc
    numeric_cols = [c for c in df.columns if c not in ()]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_px(series: pd.Series) -> pd.Series:
    return series.where(~series.isin(PX_MISSING))


def winsorize(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def main():
    df = load_raw()
    df["PX1"] = clean_px(df["PX1"])
    df["PX5"] = clean_px(df["PX5"])

    # QUAN TRỌNG: "ID" chỉ là số thứ tự phỏng vấn TRONG TỪNG THÁNG (không unique toàn bộ
    # lịch sử 1978-nay) — ghép chỉ theo ID sẽ gây match nhiều-nhiều bùng nổ tổ hợp.
    # Phải ghép theo cặp khóa (ID, YYYYMM) <-> (IDPREV, DATEPR) để đảm bảo khớp đúng người
    # ở đúng tháng phỏng vấn trước. Đổi tên toàn bộ cột thành _t1/_t2 TRƯỚC khi merge để
    # tránh nhầm lẫn do cơ chế suffix tự động của pandas.
    dup_keys = df.duplicated(subset=["ID", "YYYYMM"]).sum()
    print(f"Số dòng trùng khóa (ID, YYYYMM) trong dữ liệu gốc: {dup_keys}")
    df_unique_key = df.drop_duplicates(subset=["ID", "YYYYMM"], keep=False)

    wave1 = df_unique_key.rename(columns={c: f"{c}_t1" for c in df.columns})
    wave2 = df[df["IDPREV"].notna()].rename(columns={c: f"{c}_t2" for c in df.columns})

    panel = wave2.merge(
        wave1,
        left_on=["IDPREV_t2", "DATEPR_t2"],
        right_on=["ID_t1", "YYYYMM_t1"],
        how="inner",
    )
    print(f"Số cặp panel ghép được (trước khi lọc missing): {len(panel)}")

    panel["delta_sentiment"] = panel["ICS_t2"] - panel["ICS_t1"]
    panel["delta_px1"] = panel["PX1_t2"] - panel["PX1_t1"]
    panel["delta_px5"] = panel["PX5_t2"] - panel["PX5_t1"]
    panel["mode_switch"] = panel["METHOD_t2"] != panel["METHOD_t1"]
    panel["spans_2024_transition"] = (
        (panel["YYYYMM_t1"] >= 202401) & (panel["YYYYMM_t2"] <= 202407)
    )

    keep = panel.dropna(subset=["delta_sentiment", "delta_px1"]).copy()
    keep["delta_sentiment_w"] = winsorize(keep["delta_sentiment"])
    keep["delta_px1_w"] = winsorize(keep["delta_px1"])

    print(f"Số cặp panel hợp lệ (có ΔSentiment và ΔPX1): {len(keep)}")
    print(f"Số cặp bắc cầu qua giai đoạn chuyển đổi 2024 (T1..T7/2024): {keep['spans_2024_transition'].sum()}")
    print(f"Số cặp đổi phương thức khảo sát (mode_switch): {keep['mode_switch'].sum()}")

    keep.to_csv(OUT_PATH, index=False)
    print(f"Đã ghi {OUT_PATH}")


if __name__ == "__main__":
    main()
