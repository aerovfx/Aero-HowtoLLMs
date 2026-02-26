
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
# Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 2)

## Tóm tắt (Abstract)
Báo cáo này hoàn thiện thử thách tìm kiếm "nơ-ron ngôn ngữ" bằng cách áp dụng các phép kiểm định thống kê trên dữ liệu hoạt hóa đã thu thập. Thông qua kiểm định t-test mẫu cặp (paired samples t-test) và hiệu chỉnh Bonferroni cho đa so sánh, nghiên cứu xác định được các nơ-ron có sự khác biệt về kích hoạt đạt mức ý nghĩa thống kê giữa danh từ và động từ. Giai đoạn cuối của thực nghiệm kiểm chứng tính tổng quát hóa (generalizability) của kết quả trên một văn bản Wikipedia hoàn toàn mới, sử dụng bản đồ nhiệt (heatmaps) để quan sát sự tương quan định tính. Kết quả cho thấy sự tồn tại của tính chọn lọc sơ khai, đồng thời chỉ ra những hạn chế cố hữu của việc phân tích nơ-ron đơn lẻ trong các hệ thống phân tán phức tạp.

---

## 1. Mở Đầu (Introduction)
Sau khi đã thu thập được ma trận hoạt hóa thô ở Phần 1, thách thức tiếp theo là tách biệt tín hiệu thực sự khỏi nhiễu ngẫu nhiên. Trong khoa học dữ liệu, việc quan sát thấy sự khác biệt bằng mắt thường là chưa đủ; chúng ta cần một khung xác suất để khẳng định liệu nơ-ron 512 có thực sự "ưa thích" danh từ hơn động từ hay đó chỉ là sự biến thiên ngẫu nhiên của mẫu thử.

---

## 2. Phân tích Thống kê (Statistical Analysis)

### 2.1. Kiểm định T-test và Hiệu chỉnh Đa so sánh
- **Phép thử:** Sử dụng `scipy.stats.ttest_1samp` trên giá trị hiệu số (difference scores) giữa hoạt hóa danh từ và động từ. Đây là cách tiếp cận tương đương với paired t-test nhằm cô lập biến số nơ-ron.
- **Hiệu chỉnh Bonferroni:** Với 3072 nơ-ron được kiểm định đồng thời, ngưỡng ý nghĩa $\alpha = 0.05$ là quá lỏng lẻo. Ngưỡng mới được thiết lập là $\alpha_{adj} = 0.05 / 3072 \approx 1.6 \times 10^{-5}$ để kiểm soát tỷ lệ lỗi loại I.

### 2.2. Phân loại Nơ-ron
- **T-value dương:** Nơ-ron kích hoạt mạnh hơn đáng kể cho Danh từ.
- **T-value âm:** Nơ-ron kích hoạt mạnh hơn đáng kể cho Động từ.
Thực nghiệm cho thấy một tỷ lệ nhỏ nơ-ron vượt qua ngưỡng Bonferroni, chứng minh tính chuyên biệt hóa không phải là ngẫu nhiên.

---

## 3. Kiểm chứng Tính Tổng quát hóa (Generalizability Test)

### 3.1. Dữ liệu Văn bản Mới
Sử dụng một đoạn văn bản trích từ Wikipedia về chủ đề "Ngẫu nhiên" (Randomness) – một ngữ cảnh hoàn toàn khác với các từ đơn lẻ ban đầu. Mục tiêu là xem liệu nơ-ron "đỉnh" vừa tìm được có phản ứng chính xác với các danh từ/động từ xuất hiện tự nhiên trong câu hay không.

### 3.2. Trực quan hóa bằng Heatmap
Văn bản được tô màu dựa trên cường độ hoạt hóa của hai nơ-ron cực đoan nhất:
- **Nơ-ron Danh từ (Max T-value):** Các từ như "entropy", "uncertainty", "information" được tô màu đỏ đậm. Các hư từ hoặc động từ có màu nhạt.
- **Nơ-ron Động từ (Min T-value):** Các từ như "is", "applies", "follow" có mức độ kích hoạt cao hơn (màu xanh đậm).

---

## 4. Thảo Luận: Hạn chế và Hướng đi tiếp theo
Dù kết quả mang tính khích lệ, báo cáo chỉ ra các rào cản quan trọng:
1. **Sự đa nghĩa (Polysemanticity):** Một nơ-ron có thể vừa chọn lọc danh từ, vừa phản ứng với một ký tự đặc biệt như dấu chấm phẩy (;).
2. **Vấn đề ngữ cảnh (Context Gap):** LLM vốn được huấn luyện để xử lý chuỗi. Việc kiểm tra từ đơn lẻ (out-of-context) có thể không phản ánh đúng chức năng thực tế của nơ-ron trong các mạch điện (circuits) phức tạp.
3. **Tính chọn lọc tương đối:** Để khẳng định "chọn lọc danh từ", cần kiểm soát thêm nhiều từ loại khác (tính từ, trạng từ) thay vì chỉ so sánh nhị phân.

---

## 5. Kết Luận
Thử thách này minh chứng rằng các thành phần nội bộ của LLM (đặc biệt là MLP) chứa đựng những cấu trúc ngôn ngữ có thể giải mã được. Mặc dù không hoàn hảo, nhưng phương pháp Hooks kết hợp với thống kê cổ điển mở ra một lối đi hứa hẹn cho việc "đọc vị" tư duy máy móc, chuyển từ quan sát hành vi đầu ra sang hiểu biết về các biểu diễn ngôn ngữ nội tại.

---

## Tài liệu tham khảo (Citations)
1. Kiểm định thống kê và tổng quát hóa tính chọn lọc nơ-ron trên GPT-Neo dựa trên `aero_LLM_09_CodeChallenge Grammar tuning in MLP neurons (part 2).md`. Phân tích T-values và kiểm chứng qua Heatmaps trên dữ liệu Wikipedia.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
