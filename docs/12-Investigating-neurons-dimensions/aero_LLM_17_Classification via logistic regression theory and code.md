
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [12 Investigating neurons dimensions](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Hồi quy Logistic: Lý thuyết và Triển khai Phân loại Nơ-ron

## Tóm tắt (Abstract)
Báo cáo này giới thiệu phương pháp Hồi quy Logistic (Logistic Regression) – một công cụ thống kê thiết yếu trong việc dự đoán các biến mục tiêu phân loại (categorical outcomes). Trong bối cảnh Diễn giải học (Mechanistic Interpretability), phương pháp này được sử dụng để xác định khả năng dự đoán của hoạt hóa nơ-ron đối với các đặc tính ngôn ngữ (ví dụ: phân biệt Danh từ và Động từ). Chúng ta sẽ khám phá nền tảng toán học của hàm Logit, lý do ưu tiên xác suất log (log-probabilities) và quy trình chuẩn để triển khai, trực quan hóa kết quả bằng thư viện `statsmodels`.

---

## 1. Nền tảng Lý thuyết

### 1.1. Bản chất của Hồi quy Logistic
Hồi quy Logistic được sử dụng khi biến phụ thuộc (Dependent Variable - DV) mang tính nhị phân (Binary) – chỉ nhận một trong hai giá trị loại trừ lẫn nhau (ví dụ: Sống/Chết, Thắng/Thua, Danh từ/Động từ). 
- **Lưu ý:** Hồi quy Logistic không trực tiếp gán nhãn dữ liệu mà tính toán **xác suất ($p$)** một điểm dữ liệu thuộc về một danh mục cụ thể. Một ngưỡng (threshold), thường là 0.5, sẽ được áp dụng sau đó để đưa ra dự đoán cuối cùng.

### 1.2. Công thức Toán học
Mô hình hồi quy được thiết lập để dự đoán log-odds (logarit của tỷ lệ xác suất):
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n$$
Chuyển đổi để tìm xác suất $p$:
$$p = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \dots)}}$$
Hàm này (hàm Sigmoid) nén mọi giá trị đầu vào vào khoảng $(0, 1)$, phù hợp với định nghĩa của xác suất.

### 1.3. Tại sao sử dụng Log-Probabilities?
1. **Dải giá trị (Range):** Xác suất bị giới hạn trong $[0, 1]$, trong khi log-probabilities có dải giá trị rộng hơn, giúp mô hình hóa dễ dàng hơn.
2. **Ổn định số học (Numerical Stability):** Các xác suất cực nhỏ gần bằng 0 có thể gây ra lỗi dưới mức (underflow) trong tính toán. Log-probabilities giải quyết vấn đề này bằng cách chuyển phép nhân xác suất thành phép cộng log.

---

## 2. Quy trình Thực nghiệm trên Dữ liệu Giả lập

### 2.1. Khởi tạo và Trực quan hóa (Jittering)
Chúng ta mô phỏng hoạt hóa của nơ-ron cho hai nhóm:
- **Danh từ (Nouns):** Phân phối Gaussian với $\mu=0$.
- **Động từ (Verbs):** Phân phối Gaussian với $\mu=2$.
Kỹ thuật **Jittering** (thêm nhiễu ngẫu nhiên vào trục X) được sử dụng để tránh chồng lấp các điểm dữ liệu, giúp quan sát rõ mật độ phân phối.

### 2.2. Xây dựng Ma trận Thiết kế (Design Matrix)
Để mô hình hóa chính xác, chúng ta cần:
- Vector dữ liệu độc lập (biên độ hoạt hóa).
- Vector biến phụ thuộc (nhãn 0 và 1).
- **Hằng số (Intercept):** Sử dụng `sm.add_constant` để thêm một cột toàn giá trị 1 vào ma trận thiết kế, cho phép mô hình xử lý các trường hợp dữ liệu không có trung bình bằng 0.

### 2.3. Khớp mô hình với `Statsmodels`
Sử dụng `sm.Logit(y, X).fit()` để tìm các hệ số $\beta$ tối ưu. Bảng tóm tắt kết quả (`summary()`) cung cấp:
- **Coefficient (Hệ số):** Dấu của hệ số chỉ ra hướng ảnh hưởng (dương: hoạt hóa mạnh dự báo nhãn 1; âm: hoạt hóa mạnh dự báo nhãn 0).
- **P-value ($P>|z|$):** Kiểm định ý nghĩa thống kê của nơ-ron đối với bài toán phân loại.

---

## 3. Đánh giá Hiệu năng: Độ chính xác (Accuracy)

Sau khi có xác suất dự đoán từ `result.predict()`, chúng ta so sánh với nhãn thực tế theo ngưỡng 0.5:
$$\text{Accuracy} = \frac{\text{Số dự đoán đúng}}{\text{Tổng số mẫu}}$$
Thực nghiệm cho thấy ngay cả khi có sự trùng lắp (noise) giữa hai phân phối, Hồi quy Logistic vẫn trích xuất được ranh giới quyết định (decision boundary) tối ưu để tối đa hóa khả năng phân loại của nơ-ron.

---

## 4. Kết Luận
Hồi quy Logistic cung cấp một khung làm việc khắt khe hơn so với kiểm định T-test đơn thuần, cho phép chúng ta không chỉ xác định sự khác biệt mà còn định lượng khả năng "đọc hiểu" danh mục của từng nơ-ron đơn lẻ. Đây là bước đệm quan trọng để tiến tới phân tích nơ-ron trên dữ liệu thực của mô hình ngôn ngữ.

---

## Tài liệu tham khảo (Citations)
1. Lý thuyết và thực hành Hồi quy Logistic trên nơ-ron dựa trên `aero_LLM_17_Classification via logistic regression theory and code.md`. Triển khai với thư viện Statsmodels và phân tích độ chính xác.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
