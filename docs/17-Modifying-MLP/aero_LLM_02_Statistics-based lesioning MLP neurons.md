
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [17 Modifying MLP](../index.md)

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
# Cắt bỏ Tiệm cận các Neurons MLP trên cơ sở Thống kê (Statistics-based Lesioning of MLP Neurons)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu một phương pháp luận tiên tiến để xử lý "Sự bùng nổ chiều" của khối MLP trong các LLM bằng cách sử dụng thống kê suy diễn. Thay vì dựa trên các đặc tính mô tả đơn thuần, nghiên cứu sử dụng phép thử T-test trên một tập dữ liệu độc lập (Him/Her sentences) để xác định các neurons có phản ứng đặc hiệu với giới tính. Những neurons này sau đó được "cắt bỏ" (lesioned) có mục tiêu trong một tác vụ dự đoán token bị che khuất (masked token prediction). Kết quả thực nghiệm trên mô hình BERT chứng minh rằng việc vô hiệu hóa chỉ một nhóm nhỏ các neurons được chọn lọc theo thống kê có thể làm suy yếu khả năng nhận dạng ngữ pháp của mô hình, đồng thời cung cấp các bằng chứng thực nghiệm về tính phân tách chức năng của MLP neurons.

---

## 1. Mở Đầu (Introduction)
Việc tìm kiếm "cây kim trong đống cỏ" – tức là các neurons mang thông tin cụ thể trong số hàng ngàn đơn vị của lớp MLP – yêu cầu các công cụ sắc bén hơn là chỉ quan sát hoạt hóa thô. Nghiên cứu này đề xuất một quy trình ba phần: (1) Nhận dạng neurons đặc hiệu qua T-test; (2) Thiết lập baseline dự đoán ngữ pháp; (3) Thực hiện can thiệp nhân quả để kiểm chứng vai trò của các neurons đã chọn.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Phần 1: Nhận dạng Neurons đặc hiệu (Independent Selection)
- **Mục tiêu:** Tìm neurons phân biệt giữa "him" và "her".
- **Kỹ thuật:** Sử dụng 108 câu mẫu (54 cặp). Trích xuất hoạt hóa tại tầng `intermediate.dense` (trước khi qua hàm kích hoạt phi tuyến).
- **Phân tích:** Chạy 3072 phép thử T-test (một lần cho mỗi neuron). Áp dụng hiệu chỉnh FDR (False Discovery Rate) để kiểm soát lỗi đa so sánh.
- **Phân loại:** Neurons có $T > 0$ và $p < 0.05$ được gọi là "Him neurons", ngược lại là "Her neurons".

### 2.2. Phần 2: Nhiệm vụ dự đoán Masked Token
- **Câu mẫu:** "Robert helped Lucy with her project and she thanked him for his hard work."
- **Kịch bản:** Che khuất từ "her" hoặc "him" và quan sát xác suất logit của mô hình Bert. Đây là phép thử về khả năng hiểu cấu trúc ngữ pháp và quan hệ thực thể.

### 2.3. Phần 3: Can thiệp Nhân quả (Lesioning)
- **Thực hiện:** Sử dụng Forward Hook để gán giá trị 0 cho các neurons đã được xác định từ Phần 1 tại đúng vị trí token bị che khuất.
- **Vị trí can thiệp:** Tầng `intermediate` (đầu ra của hàm kích hoạt GELU). Việc can thiệp ở đây hay ở `dense` đều cho kết quả tương đương vì $f(0) = 0$.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Tính thưa (Sparsity) và Hàm GELU
Quan sát biểu đồ Histogram của hoạt hóa:
- **Phân phối:** Các giá trị tiền kích hoạt phân phối chuẩn (Gaussian), thuận lợi cho phép thử T.
- **Tác động của GELU:** Hàm kích hoạt này triệt tiêu phần lớn các giá trị âm, tạo ra tính "thưa" (sparsity) cho lớp MLP. Điều này giải thích tại sao chỉ một nhóm nhỏ neurons thực sự đóng góp vào residual stream tại một thời điểm nhất định.

### 3.2. Hiệu ứng của việc cắt bỏ (Lesioning Effect)
- **Logit Difference:** Khi cắt bỏ các "Her neurons" tại vị trí mask của từ "her", sai lệch logit cho từ "her" giảm xuống so với mô hình sạch. Tương tự với "him".
- **Sanity Check:** Các logits ở câu không bị can thiệp (Clean sentence) giữ nguyên mức 0, xác nhận Hook hoạt động chính xác và có mục tiêu.
- **Độ tinh vi:** Hiệu ứng quan sát được là nhỏ nhưng nhất quán. Điều này là hợp lý vì chúng ta chỉ tác động lên một tập hợp con neurons tại duy nhất một Transformer Block.

---

## 4. Thảo Luận: Thống kê trong Token-level Interventions
Một điểm mới trong phương pháp này là việc biến mô hình ngôn ngữ thành một "máy chạy thống kê". 
- **Dynamic Hooks:** Việc đưa logic T-test vào trong Hook cho phép mô hình tự thực hiện các phân tích phức tạp ngay trong Forward Pass.
- **Functional Separation:** Kết quả củng cố niềm tin rằng MLP không chỉ là các "bộ nhớ" tĩnh mà còn chứa các mạch logic chuyên biệt cho việc xử lý các đặc điểm ngôn ngữ như giới tính hay quan hệ thực thể.

---

## 5. Kết Luận
Báo cáo đã chứng minh tính hiệu quả của việc kết hợp thống kê suy diễn vào diễn giải học cơ học. Phương pháp này cho phép chúng ta không chỉ quan sát mà còn điều khiển được dòng chảy thông tin trong LLM một cách tinh vi. Những bước tiếp theo sẽ bao gồm việc mở rộng can thiệp lên nhiều tầng đồng thời để quan sát hiệu ứng cộng dồn thảm khốc (catastrophic interference).

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm Statistics-based Lesioning trên BERT dựa trên `aero_LLM_02_Statistics-based lesioning MLP neurons.md`. Phân tích neurons đặc hiệu giới tính và tác động logit.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
