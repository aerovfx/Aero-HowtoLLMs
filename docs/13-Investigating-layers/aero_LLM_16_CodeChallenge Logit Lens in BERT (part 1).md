
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [13 Investigating layers](../index.md)

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
# Thử Thách Lập Trình (Code Challenge): Ứng Dụng Logit Lens Trong Mạng BERT (Phần 1)

## Tóm tắt (Abstract)
Thực nghiệm này khởi tạo lộ trình tích hợp phép soi tiêu cự Logit Lens lên mô hình Ngôn ngữ có tính hai chiều (Bidirectional) – BERT Large. Quá trình xử lý khởi động với thuật toán khuyết từ Masked Language Modeling (MLM). Bài nghiên cứu xoay quanh việc kiểm duyệt kiến thức nền trước khi áp dụng Logit Lens, thực hành đánh giá Z-Score trên Logits điểm cuối cùng (Final layer) để đo độ kiên định của hàm dự đoán từ bị che (Masked Token). Số liệu chỉ ra: Mô hình duy trì độ lệch chuẩn lên tới cực hạn $Z > 10$ để khẳng định lựa chọn đáp án phù hợp với hoàn cảnh. 

---

## 1. Mở Đầu (Introduction)
Dòng đời của mô hình GPT-2 giải mã tuần tự từ trái qua phải (Causal Language Model), khác biệt hoàn toàn với BERT – kỹ thuật đào tạo đọc chuỗi mã hóa kết nối đồng thời từ gọng kìm 2 đầu (Bidirectional Encoder). Ở chương trước, Logit Lens đã tỏ ra rất trơn tru với GPT-2. Nhưng khi ta muốn áp dụng việc "Nội soi Tầng ẩn" lên kiến trúc 24 khối khổng lồ như BERT-Large, cấu trúc đầu kết nối giải mã (Decoder head) sẽ đặt ra nhiều chướng ngại vật về phương trình toán.
Báo cáo Phần 1 chuẩn bị tiền đề dữ liệu và chạy đánh giá quy chuẩn về khả năng Masked Token Prediction.

---

## 2. Tiền Xử Lý: Mô Hình Và Nhiệm Vụ Khuyết Viết (Methodology)

### 2.1. Thiết Lập BERT Large Uncased
Trọng số tham chiếu sử dụng: `bert-large-uncased`. So sánh với bản Base, bản này sở hữu 24 Transformer Layers, kích thước luồng Embedded 1024 dimensions. Hệ nhúng chữ không phân biệt định hình viết hoa hay viết thường (Uncased).

### 2.2. Kiểm Thử Masked Language Model (MLM)
Thay vì sinh chữ dự đoán cuối chuỗi, thuật toán khai báo biến chèn ngang `[MASK]` (ID=103 tokenizer) vào một vị trí trung tâm.
Dữ liệu đầu vào:
> "the way you do anything is the [MASK] you do everything"
Việc xử lý Forward pass được giao phó cho GPU. Cuối cùng, vector logit điểm chốt (Final Output Logits) tại vị trí Index `[MASK]` đi vào CPU để xử lý.

---

## 3. Khảo Sát Đánh Giá (Analysis)

### 3.1. Truy Vấn Argmax Khớp Tín Hiệu (Max Logit Alignment)
Hàm mục tiêu được trích xuất bằng cách định vị Argmax cao nhất bên trong không gian 30,522 tokens của thư viện BERT vocabulary. Mô hình tính toán và trả về chuỗi đích giải mã: "way" -> Hoàn thành câu "the way you do anything is the way you do everything". BERT thể hiện độ chính xác tuyệt đối nhờ khả năng trau chuốt từ 2 hướng.

### 3.2. Hiệu Chỉnh Đầu Ra Thành Z-Score (Z-Score Standardization)
Thay vì sử dụng Logits thô hoặc Softmax phi tuyến tính, ta làm phẳng định danh toàn bộ mảng Logit 30,522 bằng phân phối chuẩn Standardized Normal Distribution (Z-Score):
$$ Z_i = \frac{X_i - \mu_{vocab}}{\sigma_{vocab}} $$
Đồ thị phân vạch vạch trần ưu thế cực trị của BERT: Từ ngữ được dự đoán "way" bắn vọt lên biên độ $Z > 10$ (10 độ lệch chuẩn). Tính năng Z-score không những khử độ chệch độ lớn tự do của các LLM, mà còn đảm bảo chắc chắn rằng đối với một ngữ cảnh đúng đắn, mô hình sẽ dồn toàn bộ lực chú ý kéo cách biệt Token đáp án ra thật xa khởi nhiễu thông dụng của đại từ vựng. 

---

## 4. Bàn Luận Tạm Thời
Thao tác thay thế Masked Token Prediction hoạt động như một cỗ máy hoàn hảo tại khối Layer 24 cuối cùng. 
Tuy nhiên, cấu tạo hàm phân giải đầu ra của BERT không phải là Single Linear Matrix. Kỹ xảo Logit Lens cơ bản áp dụng lên GPT-2 sẽ trở nên vô giá trị nếu đối chiếu nhầm với cấu trúc module giải mã phức hợp "Predictions" của mạng BERT. Hiện tượng này sẽ được tháo gỡ tại phần hai của bài viết.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm đo lường Z-Score trích xuất từ dữ liệu `aero_LLM_16_CodeChallenge Logit Lens in BERT (part 1).md` (Thiết lập BERT Large, Masking cơ bản, và tiêu chuẩn hóa điểm dự đoán Logit theo Standard Deviation).
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
