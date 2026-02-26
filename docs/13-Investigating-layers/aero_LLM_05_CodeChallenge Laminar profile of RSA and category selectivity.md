
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
# Khảo Sát Phân Tầng (Laminar Profile) Về RSA Và Sự Chọn Lọc Phân Nhóm 

## Tóm tắt (Abstract)
Thực nghiệm này lập quy trình tự động mở rộng kỹ thuật chỉ số RSA (Representational Similarity Analysis) và Chọn Lọc Phân Nhóm (Selectivity Index/Category Code) diễn ra ở File 04 lên toàn bộ 36 tầng biến đổi (transformer blocks) của mô hình GPT-2 Large. Thay vì chỉ khảo sát $Q$ và $K$, tiến trình cũng quét chi tiết trên ma trận $V$. Phân tích laminar cho thấy sức mạnh mã hóa theo hạng mục ngữ nghĩa tập trung chủ yếu ở không gian kiến trúc nông (early layers) và biến hóa thành "nụ cười" phục hồi ở cụm Giá Trị ($V$) ở các tầng sâu nhất.

---

## 1. Mở Đầu (Introduction)
Với những bằng chứng về việc phân cụm ngữ nghĩa ở một tầng cố định, ta có quyền đặt ra giả thuyết về toàn bộ dòng chảy hoạt động xuyên chiều dọc mô hình $\rightarrow$ một "laminar profile" hoàn chỉnh.
Ở phương pháp này, chúng tôi quan sát mô hình GPT-2 Large nhằm đo đạc biến thiên hình thái mạng Q, K, V thông qua hàng loạt các đầu vào mục tiêu như phân nhóm Vũ trụ, Nội thất và Trái cây. Quá trình tính độ tương quan giữa các ma trận khoảng cách tương đồng diễn ra xuyên suốt ở cả 3 chiều không gian cấu trúc, từ đó làm bật lên những định hướng mã hóa nội bộ đặc sắc của ngôn ngữ theo tầng sâu (depths).

---

## 2. Phương Pháp Chuyên Biệt (Methodology)

### 2.1. Nâng Cấp Kích Thước GPT-2 Large (36 Blocks)
Kịch bản tương tự khi dùng mẫu câu *"The next word is [Target Word]"*, kết xuất 34 dòng suy luận cho 34 token. 
Mỗi block $(1 \dots 36)$ đều xuất tensor 3 chiều: Batch $\times$ Number Tokens $\times$ Dimensionality $(34 \times 5 \times 1280)$. Trích xuất phần cuối (final token).

### 2.2. Tiến Trình Phân Tính Liên Lớp (Layer Loop Computation)
Để xây dựng biểu đồ hình nón (Laminar Plot):
- Tại Layer $i$, sinh ba ma trận $34 \times 34$ cho tính chất Tương Đồng Cosine (Cosine Similarity) ứng với $Q$, $K$ và $V$.
- Chạy mặt nạ lọc để tính điểm số `Selectivity Index` riêng rẽ 3 mảng (nhóm Vũ Trụ, Nội Thất, Trái Cây).
- Thực thi tiếp hàm tương quan `Pearson Correlations` (đầu ra của tiến trình RSA).
- Ghi nhận tất cả thành ma trận lớn (Kích cỡ: 36 Layers $\times$ 3 Metrics $\times$ 3 Components Q/K/V).

---

## 3. Khám Phá Khối Dữ Liệu Lớp Ngang (Analysis & Visualizations)

### 3.1. Sự Ổn Định Điển Hình Của Chỉ Số RSA
Kết quả vẽ Scatter Plot tuyến tính (Transformer layer trên x-axis) cung cấp thông tin ấn tượng: Dù lớp nông hay sâu, **Chỉ số tương đồng đại diện (RSA)** giữa những Token vựng vẫn duy tri mức từ khá cao đến cực cao $(0.84 \to 0.96)$. Điều này mang ý nghĩa: Các tổ hợp mã biểu diễn và nhận diện mối hệ thuộc giữa các nhóm từ hoạt động rất cứng cáp xuyên chiều không gian hệ thống.

### 3.2. Đường Biến Đổi Hàm Chọn Lọc Mục Tiêu (Selectivity Dynamics)
Ngược lại với sự đi ngang của tính tương đồng RSA:
- **Tầng Nông (Early Layers):** Các không gian có một bước nhịp `Selectivity Index` rất cao cho cấu trúc $Q$ và $K$. Bởi vì các Layers ngoài cùng kề cận sát ma trận Embeddings, chúng đảm đương việc phác thảo định kiến "vật lý" gần nghĩa của từ nhất. Tại đây tính chất Category cực thịnh.
- **Tầng Giữa & Sâu (Middle to Deep Layers):** Mức Selectivity dần phân ra và sa sút. Mô hình không còn quan tâm nhiều câu chuyện đây là từ thuộc nhóm Vũ trụ hay Trái cây, mục đích tiên quyết ngả dần về dự đoán hàm tiếp điểm sau cùng (prediction-oriented context processing).
- **Cú Bẻ Lái Của Chùm Giá Trị (The Smile Pattern in V Matrix):** Một phát hiện đầy kinh ngạc ở Layer $> 18$ cho thấy ma trận $V$ đánh dấu sự hồi sinh của đồ thị hình học nụ cười (Smile Pattern). Sức mạnh gộp nhóm ngữ nghĩa ở riêng mạng Giá Trị đột ngột tăng theo đường tiệm cận, có lẽ là quá trình nó đóng gói các thuộc tính ẩn để rải rác phân phát về $Residual\ Stream$ theo cụm.

---

## 4. Kết Luận (Conclusion)
Thông qua kỹ thuật lặp `Loop Matrix` đối lưu xuyên suốt các tầng kiến trúc Transformer, chúng tôi vẽ đồ thị Laminar về sức sống của 3 nhóm Semantic Vocabulary. Quá trình tính năng giải cơ học bộc lộ rằng: Quỹ đạo định tuyến phân loại (Categorical coding) đạt cực đại tại khu vực trạm thu phát tín hiệu ban đầu (Input Embeddings Stage). Bắt đầu từ đoạn giữa tới đuôi hành trình, ý niệm về "Phân nhánh" bị lu mờ trước tác động "Dịch chuyển bối cảnh", chỉ trừ quá trình đặc thù trên phân khúc $V$ - phục vụ khả năng giao kết nội dung tổng hợp.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh thí nghiệm liên kết: `aero_LLM_05_CodeChallenge Laminar profile of RSA and category selectivity.md` (Giải mã sự thay đổi đặc tính Selectivity Indexes trên 36 tầng Layers của Model GPT-2 Large).
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
