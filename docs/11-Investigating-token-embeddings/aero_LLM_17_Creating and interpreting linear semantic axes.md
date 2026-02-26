
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [11 Investigating token embeddings](../index.md)

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
# Thiết Lập Và Diễn Giải Trục Ngữ Nghĩa Tuyến Tính (Linear Semantic Axes)

## Tóm tắt

Các chiều trong không gian nhúng của hệ mô hình tự hồi quy lớn (Autoregressive LLMs) thường được gán cho một tính chất thần bí khi mà các nhà khoa học có thể cộng trừ các đại lượng định danh để tìm các góc độ ngữ pháp (VD: Vector Tương lai - Vector Quá khứ = Trục thời gian). Bài báo khoa học này chứng minh tầm ảnh hưởng của thao tác Chuẩn hóa hình học (Normalization) đối chiếu khả năng sàng lọc tín hiệu với một bộ lọc thô sơ trong quá trình làm nét "Trục ngữ nghĩa tuyến tính" của cụm nhúng.

---

## 1. Trục Ngữ Nghĩa: Đường Vẽ Logic Chạy Xuyên Ma Trận

Không gian từ vựng Word2Vec, theo lý thuyết, chứa khả năng biểu diễn những khái niệm tương phản ở hai phía của một đường thẳng. Giả sử ta muốn xác lập một **Trục Thời Gian (Time Axis)**, phép tính lấy điểm nút (anchor points) là hai tọa độ đặc trưng đối lập "Past" và "Future":
$$
\vec{v}_{\text{TimeAxis}} = \vec{v}_{\text{future}} - \vec{v}_{\text{past}}
$$
Một khi đã xác lập được $\vec{v}_{\text{TimeAxis}}$, mọi vector nhúng $\vec{w}$ bất kỳ khi chiếu (project) lên trục này sẽ trả về hệ số (projection scalar) dự đoán mức độ "thuộc về tương lai" hay "hoài niệm quá khứ" thông qua phép Tích vô hướng (Dot product).

---

## 2. Tiền Xử Lý Hình Học (Geometric Pre-Normalization)

Cạm bẫy tiềm ẩn của việc trừ đi hai tọa độ thô nằm ở "Sức nặng vi phân" của mỗi token. Những từ vựng thông thường nhưng vô nghĩa (stop words như "the", "an", "is") chứa vector embeddings mờ với chiều dài chuẩn ngắn (low $L2-norm$). Ngược lại các từ ngữ cảnh trọng điểm sẽ có $\vec{v}$ với chiều dài cực đại đâm xa khỏi gốc tọa độ $0$.

Nếu ta lấy $\vec{v}_{\text{future}} - \vec{v}_{\text{the}}$, đáp án sẽ bị nghiêng lệch (bias) khổng lồ về phía đầu điểm "future" khiến cho trục không gian thành phẩm bị trượt góc mất tính đối xứng tương sinh. Vấn đề được giải quyết bằng việc bắt buột **Chuẩn Hóa (Normalization)** độ dài từng thành phần trước khi thực hiện quy đổi trục:
$$
\hat{v}_{\text{future}} = \frac{\vec{v}_{\text{future}}}{\|\vec{v}_{\text{future}}\|}
$$
$$
\hat{v}_{\text{past}} = \frac{\vec{v}_{\text{past}}}{\|\vec{v}_{\text{past}}\|}
$$
Trục ngữ nghĩa thực thụ (Normalized Axis) phải được thiết lập trên hai vector chuẩn quy có độ dài giới hạn trong vòng viền cầu bằng $1$:
$$ 
\vec{v}_{\text{TimeAxisNorm}} = \hat{v}_{\text{future}} - \hat{v}_{\text{past}} 
$$
Tính khưỡng bức không gian này tước đi ảo ảnh phương sai từ độ lớn module, khiến hệ quy chiếu chỉ tập trung vào khác biệt phương hướng góc (Cosine direction divergence).

---

## 3. Hệ Quả Từ Những Bộ Lọc Căn Bản (Tokens Filtering Rule)

Khi tiến hành chấm điểm (Cosine similarity mapping) một "Trục ranh giới thời gian" với một bộ từ điển lên đến hàng triệu từ vựng cắt ra từ Wikipedia, một số kết quả lạ lẫm âm cực có thể nổ ra (những liên kết token nhiễu như địa chỉ URL, ký tự lỗi, chữ Latin viết tắt trộn lẫn điểm ngẫu nhiên). Để khử các yếu tố nhiễu này, logic Lọc nhãn từ vựng (Filters) được bổ sung:
- **Chuẩn Cự ly Chữ cái:** Từ vựng yêu cầu $> 2$  letters.
- **Biểu thức chính quy (Alphanumeric Mask):** Chỉ những mảng kí tự hoàn toàn cấu tạo từ chữ hệ ABC.

Bộ lọc này quét qua ma trận từ nhúng $E \in \mathbb{R}^{V \times D}$ (với $V=3.000.000$). Kết quả loại bỏ đến $70\%$ khối lượng vocab của Word2vec chỉ chứa rác thông tin (noise artifacts). Mảng rút lõi giúp tăng tốc đạo hàm bộ nhớ trên $E_{filtered}$, cho phép $\text{CosSim}(\vec{v}_{\text{axis}}, E_{filtered}^T)$ tìm trúng những vạch đích từ vựng ("pass performance", "yesterday") nằm đúng phổ âm của trục mà không dính bẫy ngẫu nhiên (false positive correlation).

---

## 4. Kết luận

Các thực thể LLMs với biểu đồ Attention không hoạt động bằng pháp thuật - chúng là tổ hợp băm rã những lớp Norm (Normalization) xếp chồng nhau và những thủ thuật màng lọc nhỏ cấu thành một kiến trúc phi tuyến kỳ vĩ. Tính toán các điểm lân cận ngữ nghĩa trên mạng Nơ-ron yêu cầu sự nghiêm ngặt định chuẩn (vector normalizations) nhằm không để cho tính đa dạng ngẫu nhiên của khối lượng tọa độ chiễm lĩnh độ cân bằng tuyến tính của hình học ngôn ngữ học.

---

## Tài liệu tham khảo

1. **Bolukbasi, T., et al. (2016).** *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. NeurIPS. (Áp dụng trục ngữ nghĩa khử thiên vị giới tính).
2. **Li, Y., et al. (2015).** *Representation Learning for Semantic Composition*. ACL.
3. Tài liệu bài giảng *Investigating token embeddings - Creating and interpreting linear semantic axes*.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
