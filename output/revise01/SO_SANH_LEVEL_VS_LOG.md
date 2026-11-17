# So sánh thang level và thang log (bootstrap 2.000 lần, cụm theo người trả lời)

Level: thay đổi kỳ vọng tính bằng điểm phần trăm. Log: 100·Δln(1+PX1/100), kỳ vọng ban đầu cũng đổi sang log. Sentiment giữ nguyên. Ở thang log, ±2 và ±5 là điểm log, p0 = 3%, 4%, 6% được đổi sang điểm log tương ứng.

## 1. Chênh lệch bất đối xứng web − tele (Common support + overlap weighting, bootstrap toàn quy trình)

| Thang | Ước lượng | CI 95% | p bootstrap | Tỷ lệ draw < 0 |
|---|---|---|---|---|
| Level | -0.482 | [-0.977, -0.007] | 0.049 | 97.6% |
| Log | -0.430 | [-0.966, 0.098] | 0.106 | 94.7% |

Bản thầy (level, 500 draw): -0.493, [-0.971, -0.037], p = 0.028. Với 2.000 draw, khoảng tin cậy level chỉ vừa chạm ngưỡng 5%.

## 2. Chênh lệch độ dốc tăng − giảm (marginal slopes, 2.000 draw)

| p0 | |Δ| | Level: chênh lệch [CI 95%] | p | Log: chênh lệch [CI 95%] | p |
|---|---|---|---|---|---|
| 3% | 2 | -0.321 [-0.625, -0.007] | 0.044 | -0.310 [-0.631, 0.021] | 0.073 |
| 3% | 5 | -0.300 [-0.491, -0.091] | 0.004 | -0.346 [-0.541, -0.132] | 0.001 |
| 4% | 2 | -0.309 [-0.612, 0.011] | 0.053 | -0.296 [-0.617, 0.042] | 0.093 |
| 4% | 5 | -0.288 [-0.483, -0.074] | 0.007 | -0.332 [-0.529, -0.117] | 0.001 |
| 6% | 2 | -0.285 [-0.598, 0.038] | 0.091 | -0.268 [-0.596, 0.077] | 0.148 |
| 6% | 5 | -0.264 [-0.472, -0.043] | 0.016 | -0.304 [-0.514, -0.083] | 0.004 |

## 3. Bốn đặc tả phi tuyến: hệ số λ cục bộ, p(độ cong bậc 2), p(kỳ vọng ban đầu)

| Mẫu | Mô hình | λ level | λ log | p cong (lvl / log) | p ban đầu (lvl / log) |
|---|---|---|---|---|---|
| Pooled | Baseline | 2.325 | 2.107 | | |
| Pooled | Bậc 2 | 2.022 | 2.160 | 0.196 / 0.111 | |
| Pooled | Ban đầu | 1.807 | 2.147 | | 0.295 / 0.591 |
| Pooled | Cả hai | 1.840 | 1.640 | 0.367 / 0.275 | 0.208 / 0.242 |
| Tele | Baseline | 1.407 | 1.258 | | |
| Tele | Bậc 2 | 2.061 | 2.118 | 0.0055 / 0.0009 | |
| Tele | Ban đầu | 2.198 | 3.649 | | 0.408 / 0.655 |
| Tele | Cả hai | 2.367 | 1.996 | 0.0095 / 0.0028 | 0.300 / 0.082 |
| Web | Baseline | 2.870 | 2.151 | | |
| Web | Bậc 2 | 1.613 | 1.801 | 0.266 / 0.473 | |
| Web | Ban đầu | 0.809 | 0.816 | | 0.0237 / 0.105 |
| Web | Cả hai | 0.687 | 0.685 | 0.122 / 0.306 | 0.0054 / 0.044 |

Lưu ý: các λ ngoài dòng Baseline là tỷ lệ hệ số cục bộ tại x = 0 và p0 = 0, không so sánh trực tiếp với baseline.

## 4. Hạn chế theo độ lớn thay đổi, mẫu web

| Mẫu | n level | λ level (p) | n log | λ log (p) |
|---|---|---|---|---|
| ≤5 | 2648 | 0.756 (0.551) | 2599 | 1.009 (0.980) |
| ≤10 | 3553 | 1.448 (0.320) | 3493 | 1.707 (0.082) |
| ≤15 | 3905 | 1.989 (0.054) | 3902 | 1.889 (0.067) |
| ≤20 | 4309 | 3.377 (<0.001) | 4495 | 2.151 (<0.001) |
| Toàn bộ web | 4495 | 2.870 (<0.001) | 4495 | 2.151 (<0.001) |

Ở thang log, mọi quan sát web đều có |Δ| ≤ 20 nên dòng ≤20 trùng toàn bộ web.

## Đọc nhanh

- Kết luận định tính không đổi: bất đối xứng phụ thuộc độ lớn thay đổi, ±5 mạnh hơn ±2, và khác biệt tele/web co lại sau khi điều chỉnh.
- Log làm yếu thêm phần khác biệt tele/web: p bootstrap 0.106 (log) so với 0.049 (level), khoảng tin cậy log chứa 0.
- λ web toàn mẫu giảm từ 2.87 xuống 2.15 khi đổi sang log.
- Độ cong bậc 2 ở telephone mạnh hơn trên thang log (p = 0.0009).
- Các phép kiểm định này đều mang tính thăm dò, và mode vẫn đồng biến với period.
