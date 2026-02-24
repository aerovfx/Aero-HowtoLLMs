# Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)

## Tóm tắt (Abstract)
Bài viết này điều tra các cơ chế biểu diễn của mạng nơ-ron ngôn ngữ lớn (LLMs), cụ thể là mô hình GPT-2, ở cấp độ tầng ẩn (layer-level). Trọng tâm nghiên cứu là phân tích mối quan hệ giữa các vector kích hoạt Truy vấn (Query - $Q$), Khóa (Key - $K$) và Giá trị (Value - $V$) dựa trên hệ số tương quan (Correlation) và độ tương đồng Cosine (Cosine Similarity). Thông qua phương pháp cô lập token đích định sẵn (từ "her") trong các ngữ cảnh câu khác biệt, chúng tôi phát hiện ra những quy luật phân bố tương đồng hội tụ rất mạnh trong không gian biểu diễn cơ cấu mạng Attention.

---

## 1. Mở Đầu (Introduction)
Phân tích mô hình ngôn ngữ lớn ở cấp độ các tầng ẩn (layer-level) cung cấp một cái nhìn tổng quan về cách thông tin được tổ chức và xử lý theo từng khối cấu trúc, cao hơn so với việc nghiên cứu từng nơ-ron (neuron) rời rạc. 

Bằng việc tìm hiểu cách biểu diễn token (token embeddings) biến đổi và tương tác qua các ma trận Tự chú ý (Self-Attention sublayers), ta có thể giải mã dần cơ chế nắm bắt ngữ cảnh của mô hình. Trong bài nghiên cứu này, chúng tôi đi sâu vào việc đối chiếu các đa tạp biểu diễn nội tại trong $Q$, $K$, $V$ khi một token hoàn toàn giống nhau được truyền qua các quy trình văn cảnh ngữ pháp (context) khác nhau.

---

## 2. Phương Pháp Thực Nghiệm & Đo Lường (Methodology)

### 2.1. Thiết Kế Tập Dữ Liệu và Bối Cảnh
Thực nghiệm sử dụng mô hình GPT-2 (small), tiến hành trích xuất hàm kích hoạt (hooking activations) thẳng từ vòng Transformer. Một hệ thống bao gồm $54$ câu văn ngắn được đưa vào mạng.
- Cấu trúc chung: Mọi câu đều chứa chung một *token đích* cố định (ví dụ: chuỗi `[space] her`). Do đó, bản sắc cốt lõi của token đích là hoàn toàn giống hệt nhau (identical) về định danh đầu vào.
- Tính độc lập: Điều làm nên khối dữ liệu đối sánh là token đứng trước/sau và tổng chiều dài mỗi chuỗi chứa token thay đổi - buộc hệ thống tự padding tự động. 

### 2.2. Đo Lường Bằng Độ Tương Đồng Cosine (Cosine Similarity)
Để kiểm chứng ma trận vector kích hoạt nội tại, ta áp dụng công thức Độ tương đồng Cosine, định nghĩa sự trùng lặp góc đo định hướng giữa vector $x$ và vector $y$:
$$ \text{Cosine Similarity}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2}\sqrt{\sum_{i=1}^{n} y_i^2}} $$
*Khai triển tính toán vi mô:* Các ma trận này được ánh xạ đại số bằng cấu trúc phép nhân vô hướng của ma trận dữ liệu chuyển vị (transpose) lên chính nó, và chia cho quy chuẩn độ dài đường chuẩn $L_2$ (matrix norm) nhằm tạo ra các tập hợp phân bổ nằm giới hạn trong khung giá trị lý tưởng $[-1, 1]$.

---

## 3. Khám Phá Khối Dữ Liệu Nội Tại (Results & Analysis)

Hệ thống trích xuất Tensor xuất bản ghi với dạng biến chiều $\mathbf{54 \times 8 \times 2304}$ (Tương ứng: Khối 54 chuỗi câu $\times$ 8 tokens mặc định tính padding $\times$ tổng concat của Q,K,V vì GPT-2 small có mức $n\_embed$ là $768$).

**Phân Tích Cấu Tạo:** Hình đồ Histogram phân bố của độ tương đồng Cosine ở các chiều $Q-Q$ hoặc tương tác cặp.
- Khảo sát các kích hoạt ở từ "her" dọc theo 54 ngữ cảnh cho một kết quả kinh ngạc: Độ tương đồng Cosine của các vector đích biểu thị một hình trạng hội tụ hướng hai cực, thường là **dương rất đậm** hoặc thỉnh thoảng sẽ mang xu hướng **âm tương phản rõ rệt** (Strong Negative/Positive).
- Sự tồn tại của token đích giống nhau áp đảo ngữ cảnh khác nhau lên kích hoạt không gian, giữ các điểm scatter-plot gộp vào một hệ tương quan hệ số cao (ví dụ: tương tác $ > 0.9$ trên hệ quy chiếu chéo của các câu văn).

---

## 4. Kết Luận (Conclusion)
Thông qua thủ pháp móc nối các tầng ẩn của mô hình tại vòng lặp thứ $6$ (Layer-6), bài thực nghiệm chứng minh sự ổn định cơ học đáng lưu tâm tại $Q$, $K$, $V$ đối với nhóm token đích mang tính nguyên bản liên kết. Cả quá trình tính độ đo Cosinus tiết lộ sức mạnh duy trì ý nghĩa định dang ban đầu, đi ngược lại một số giả định về việc văn cảnh thay đổi sẽ hoàn toàn thay đổi quỹ đạo số học ẩn tàng.

Dữ liệu này cung cấp tiền đề để nghiên cứu sâu thêm về nhóm cụm chức năng học thuật trên mạng ngôn ngữ nhiều Parameter hơn.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ: `aero_LLM_01_Token-related similarities within and across Q, K, V matrices (part 1).md`. (Khảo cứu cách tính Cosinus, xây dựng kịch bản 54 câu mô phỏng token đích và quy mô Tensor GPT-2 small PyTorch / Numpy).
