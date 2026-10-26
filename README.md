# MNIST 10–20 Challenge

## Overview
Mục tiêu của cuộc thi là xây dựng mô hình phân loại số viết tay từ **10 đến 20** theo phong cách MNIST.

- **Kích thước ảnh:** `28×28` grayscale.
- **Số lượng class:** 11 lớp (`10, 11, ..., 20`).
- **Dữ liệu huấn luyện:** Các đội tự xây dựng dữ liệu huấn luyện và mô hình.
- **Tập Test:** Ban tổ chức chỉ cung cấp tập test không có nhãn để các đội thực hiện dự đoán và submit kết quả lên Kaggle.
- **Quy mô tập test:** 550 ảnh (*50 ảnh cho mỗi class*).

### Đánh giá & Leaderboard
Kết quả được đánh giá bằng chỉ số **Accuracy**. 
Leaderboard bao gồm:
- **Public Leaderboard:** 40% tập test (sử dụng để cập nhật thứ hạng trong thời gian thi).
- **Private Leaderboard:** 60% tập test (sử dụng để xác định thứ hạng cuối cùng).

> **Mục tiêu chính:** Xây dựng một pipeline có khả năng tổng quát tốt trên dữ liệu viết tay chưa từng thấy.

---

## Timeline
- **Start:** Oct 5, 2026
- **Close:** Oct 18, 2026

---

## Description
Bài toán được xây dựng dựa trên ý tưởng của tập dữ liệu MNIST nổi tiếng, nhưng mở rộng sang các số viết tay từ **10 đến 20**.

- Mỗi mẫu dữ liệu là một ảnh `28×28` grayscale, chứa đầy đủ một số thuộc một trong 11 class: `10, 11, 12, ..., 20`.
- **Không có training dataset chính thức từ ban tổ chức.** Mỗi đội cần tự xây dựng dữ liệu huấn luyện, lựa chọn phương pháp xử lý dữ liệu và mô hình phù hợp.
- Ban tổ chức cung cấp một tập test gồm 550 ảnh, với 50 ảnh cho mỗi class. Nhãn của tập test được giữ bí mật.
- Nhiệm vụ của các đội là dự đoán class cho từng ảnh test và submit kết quả theo đúng định dạng Kaggle.

---

## Evaluation
Kết quả được đánh giá bằng chỉ số **Accuracy**:

$$\text{Accuracy} = \frac{\text{Số dự đoán đúng}}{\text{Tổng số mẫu}}$$

*Ví dụ: Nếu mô hình dự đoán đúng $500/550$ ảnh:*
$$\text{Accuracy} = \frac{500}{550} \approx 90.91\%$$

---

## Submission Format
File submission phải là file CSV chứa đúng 2 cột: `id` và `label`.

```csv
id,label
0,10
1,17
2,20
3,14
```
---

## Citation
Pham Minh Duc Ngo, Anh Quan Ngo, EDABK Lab. "MNIST 10–20 Challenge." Kaggle Competition, 2026. EDABK AI. [https:/
