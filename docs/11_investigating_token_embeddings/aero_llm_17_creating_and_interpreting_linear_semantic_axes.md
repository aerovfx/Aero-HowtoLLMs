
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [11 investigating token embeddings](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Thiết Lập Và Diễn Giải Trục Ngữ Nghĩa Tuyến Tính (Linear Semantic Axes)

## Tóm tắt

Các chiều trong không gian nhúng của hệ mô hình tự hồi quy lớn (Autoregressive LLMs) thường được gán cho một tính chất thần bí khi mà các nhà khoa học có thể cộng trừ các đại lượng định danh để tìm các góc độ ngữ pháp $VD: Vector Tương lai - Vector Quá khứ = Trục thời gian$. Bài báo khoa học này chứng minh tầm ảnh hưởng của thao tác Chuẩn hóa hình học (Normalization) đối chiếu khả năng sàng lọc tín hiệu với một bộ lọc thô sơ trong quá trình làm nét "Trục ngữ nghĩa tuyến tính" của cụm nhúng.

---

## 1. Trục Ngữ Nghĩa: Đường Vẽ Logic Chạy Xuyên Ma Trận

Không gian từ vựng Word2Vec, theo lý thuyết, chứa khả năng biểu diễn những khái niệm tương phản ở hai phía của một đường thẳng. Giả sử ta muốn xác lập một **Trục Thời Gian (Time Axis)**, phép tính lấy điểm nút (anchor points) là hai tọa độ đặc trưng đối lập "Past" và "Future":

\vec{v}_{\text{TimeAxis}} = \vec{v}_{\text{future}} - \vec{v}_{\text{past}}

Một khi đã xác lập được $\vec{v}_{\text{TimeAxis}}$, mọi vector nhúng $\vec{w}$ bất kỳ khi chiếu (project) lên trục này sẽ trả về hệ số (projection scalar) dự đoán mức độ "thuộc về tương lai" hay "hoài niệm quá khứ" thông qua phép Tích vô hướng (Dot product).

---

## 2. Tiền Xử Lý Hình Học (Geometric Pre-Normalization)

Cạm bẫy tiềm ẩn của việc trừ đi hai tọa độ thô nằm ở "Sức nặng vi phân" của mỗi token. Những từ vựng thông thường nhưng vô nghĩa (stop words như "the", "an", "is") chứa vector embeddings mờ với chiều dài chuẩn ngắn (low $L2-norm$). Ngược lại các từ ngữ cảnh trọng điểm sẽ có $\vec{v}$ với chiều dài cực đại đâm xa khỏi gốc tọa độ $0$.

Nếu ta lấy $\vec{v}_{\text{future}} - \vec{v}_{\text{the}}$, đáp án sẽ bị nghiêng lệch (bias) khổng lồ về phía đầu điểm "future" khiến cho trục không gian thành phẩm bị trượt góc mất tính đối xứng tương sinh. Vấn đề được giải quyết bằng việc bắt buột **Chuẩn Hóa (Normalization)** độ dài từng thành phần trước khi thực hiện quy đổi trục:

\hat{v}_{\text{future}} = \frac{\vec{v}_{\text{future}}}{\|\vec{v}_{\text{future}}\|}

$$
\hat{v}_{\text{past}} = \frac{\vec{v}_{\text{past}}}{\|\vec{v}_{\text{past}}\|} Trục ngữ nghĩa thực thụ (Normalized Axis) phải được thiết lập trên hai vector chuẩn quy có độ dài giới hạn trong vòng viền cầu bằng 1: \vec{v}_{\text{TimeAxisNorm}} = \hat{v}_{\text{future}} - \hat{v}_{\text{past}}
$$

