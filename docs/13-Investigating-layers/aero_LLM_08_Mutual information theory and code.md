
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
# Lý Thuyết Thông Tin: Đo Lường Entropy Và Mutual Information

## Tóm Tắt (Abstract)
Nghiên cứu này trình bày cách ứng dụng cốt lõi của **Lý thuyết thông tin (Information Theory)** vào các phép đo lường thống kê cho dữ liệu hoạt động của nơ-ron: Entropy và Mutual Information (Thông tin tương hỗ). Không giống như phương sai hay hệ số tương quan vốn mang bản chất tuyến tính và chỉ phù hợp với phân phối chuẩn, hai khái niệm toán học này giúp định lượng mức độ hỗn loạn (độ bất định) và khả năng quy nạp phi tuyến giữa hai biến số liên tục. Sự so sánh tính toán thủ công qua Histogram và thư viện Scikit-learn cũng được đề cập nhằm thiết lập tiền đề đoạt giải cơ học trên các ma trận nơ-ron sau này.

---

## 1. Khái Niệm Về Entropy Trong Lý Thuyết Thông Tin

Trái ngược với "Entropy nhiệt động lực học" tập trung vào sự hỗn loạn của hệ vật lý, **Entropy Shannon** mang màu sắc của "Sự bất ngờ" (Surprise) và "Khả năng dự đoán" (Predictability). 
- Một sự kiện có xác suất bằng $1$ (như mặt trời mọc vào ngày mai) không có sự bất ngờ $\to \text{Entropy} = 0$.
- Một sự kiện như tung đồng xu (xác suất $0.5$) rất khó dự đoán $\to \text{Entropy}$ đạt cực đại.

### 1.1. Công Thức Toán Học
Dành cho một biến biến thiên ngẫu nhiên (hoặc các đặc trưng categorical/continuous bins):
$$ H(X) = - \sum_{i=1}^{n} P(x_i) \log P(x_i) $$
Do $P(x_i) \in [0, 1]$ nên hệ số logarit sẽ mang dấu âm, dấu trừ phía ngoài giúp triệt tiêu và giữ giá trị Entropy $H$ luôn dương.

### 1.2. Xử Lý Các Trùng Lặp Số Học (Numerical Errors)
Do đặc thù logarit không xác định tại mốc 0, khi thực nghiệm phân vùng histogram trên một dữ liệu nơ-ron dày đặc, nhiều bin sẽ xuất hiện giá trị $P=0$. Để khắc phục, công thức code thực tế thêm cực trị tàn dư nhỏ (epsilon $\epsilon$) vào lõi tính:
$$ H(X) = - \sum P(X) \log(P(X) + \epsilon) $$
Nếu $P=0$, $\log(\epsilon) \times 0$ vẫn sẽ triệt tiêu trở về $0$, tránh sụp đổ vòng lặp hàm hàm log.

---

## 2. Đo Lường Sự Đồng Biến: Mutual Information (MI)

Nếu cho 2 biến $X$ và $Y$, **Mutual Information - $I(X;Y)$** là tỷ trọng mức độ thông tin bạn có thể luận ra từ biến kia, thông qua việc biết biến còn lại. Khác với "Covariance" (Hiệp phương sai), biến MI đặc biệt xuất sắc trong việc tóm tắt các khuynh hướng cấu trúc hình học hỗn hợp.

### 2.1. Tiếp Cận Bằng Biểu Đồ Venn (Entropy Giao Thoa)
Có thể đo lường MI bằng cách tính toán hàm lượng Entropy nguyên bản và Entropy hợp bộ (Joint-Entropy):
$$ I(X;Y) = H(X) + H(Y) - H(X,Y) $$
Nói cách khác, nó là phần "giao nhau" của giới hạn độ bất định giữa $X$ và $Y$. 

### 2.2. Tiếp Cận Bằng Phương Trình Phân Phối Cụ Thể
$$ I(X;Y) = \sum_{x \in X} \sum_{y \in Y} P(x,y) \log \left( \frac{P(x,y)}{P(x)P(y)} \right) $$

---

## 3. Thực Nghiệm Phương Pháp Luận Và Sai Số (Methodology in Praxis)

Dữ liệu đặc tính nơ-ron là các phân phối biến liên tục (continuous arrays), không phải các danh mục (discrete). Điều này tạo ra một rào cản đo lường khi ta buộc phải ép dữ liệu về các mặt lưới tần suất 2D (2D Histograms).

1. **Sai số do thủ công chia Histograms:**
   - Khi đo lường biến mảng $x$ và biến vô định $y$ không liên kết (Tức $I = 0$ tuyệt đối theo lý thuyết), việc gom nhóm dữ liệu thủ công vào 15 bins hoặc phân tách bằng phân vị (Percentiles) vẫn trả về kết quả ảo $(I \approx 0.4 \to 0.5)$. Kết quả Histogram đính kèm một lực lượng "sai lệch tĩnh" (constant bias).
2. **Khắc phục bằng Công Cụ Cốt Lõi (Scikit-Learn Regression):**
   - Thay vì đếm điểm số theo ô, phương pháp Non-parametric Kernel Density Estimators thuộc hàm thư viện `mutual_info_regression` của Sklearn cho phép định đoán chính xác nhất dải phân bổ xác suất, giúp đẩy Mutual Information trả về trân diện ở ngưỡng xấp xỉ $0.0$.
   - **Đánh đổi:** Hàm Sklearn chạy cực lỳ chậm. Do đó ở các kiến trúc LLMs phân giải hàng tỷ thông số, ta vẫn ưu tiên Histogram Method vì thực chất độ lệch Bias luôn đi ngang tự nhiên, không làm sai khác tính đối chiếu tỷ lệ.

---

## 4. Kết Luận
Bài viết diễn giải góc nhìn định lượng mới về không gian hoạt động hệ thống. Mutual Information bộc lộ sự hữu hiệu khi bỏ qua khái niệm tuyến tính định chuẩn và chỉ quan tâm duy nhất đến "vật chất giao thoa về tính bất định" của hệ. Kỹ thuật này sẽ là mảnh ghép tiền đề cho phép bóc tách cấu trúc luồng của Network mà không hề bận tâm tới biên độ khuếch đại (Scaling factor) của từng tầng chức năng.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh thí nghiệm liên kết: `aero_LLM_08_Mutual information theory and code.md` (Cách sử dụng Histogram 2D vs Scikit Learn; xây dựng công thức Shannon Entropy và Pairwise Mutual Information tĩnh).
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)](aero_LLM_01_Token-related similarities within and across Q, K, V matrices (part 1).md) | [Xem bài viết →](aero_LLM_01_Token-related similarities within and across Q, K, V matrices (part 1).md) |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 2)](aero_LLM_02_Token-related similarities within and across Q, K, V matrices (part 2).md) | [Xem bài viết →](aero_LLM_02_Token-related similarities within and across Q, K, V matrices (part 2).md) |
| [Thử Thách Lập Trình (Code Challenge): Phân Tích Độ Tương Đồng Của Token Xuyên Suốt Các Tầng Ẩn](aero_LLM_03_CodeChallenge Token-related similarities across layers.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Token-related similarities across layers.md) |
| [Phân Tích Sự Phân Cụm và Tương Đồng Biểu Diễn (RSA) Trong Ma Trận Q và K](aero_LLM_04_Grouping and RSA in Q and K matrices.md) | [Xem bài viết →](aero_LLM_04_Grouping and RSA in Q and K matrices.md) |
| [Khảo Sát Phân Tầng (Laminar Profile) Về RSA Và Sự Chọn Lọc Phân Nhóm](aero_LLM_05_CodeChallenge Laminar profile of RSA and category selectivity.md) | [Xem bài viết →](aero_LLM_05_CodeChallenge Laminar profile of RSA and category selectivity.md) |
| [Phân Tích Số Chiều Hiệu Quả (Effective Dimensionality) Thông Qua PCA](aero_LLM_06_Effective dimensionality analysis with PCA.md) | [Xem bài viết →](aero_LLM_06_Effective dimensionality analysis with PCA.md) |
| [Thử Thách Lập Trình (Code Challenge): Khảo Sát Số Chiều Hiệu Quả Trên Pythia 2.8B](aero_LLM_07_CodeChallenge Dimensionalities in Pythia 2.3B.md) | [Xem bài viết →](aero_LLM_07_CodeChallenge Dimensionalities in Pythia 2.3B.md) |
| 📌 **[Lý Thuyết Thông Tin: Đo Lường Entropy Và Mutual Information](aero_LLM_08_Mutual information theory and code.md)** | [Xem bài viết →](aero_LLM_08_Mutual information theory and code.md) |
| [Phân Tích Thông Tin Tương Hỗ Dọc Theo Các Tầng Của Mô Hình Ngôn Ngữ (Pairwise Mutual Information Through LLMs)](aero_LLM_09_Pairwise mutual information through the LLM.md) | [Xem bài viết →](aero_LLM_09_Pairwise mutual information through the LLM.md) |
| [Phân Tích Đối Chiếu Đo Lường Tương Quan: Mutual Information và Covariance](aero_LLM_10_Mutual information vs. covariance.md) | [Xem bài viết →](aero_LLM_10_Mutual information vs. covariance.md) |
| [Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 1)](aero_LLM_11_CodeChallenge Attention to coffee MI and token distances (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Attention to coffee MI and token distances (part 1).md) |
| [Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 2)](aero_LLM_12_CodeChallenge Attention to coffee MI and token distances (part 2).md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Attention to coffee MI and token distances (part 2).md) |
| [Phân Khảo Cấu Trúc Cụm (Clusters): Dấu Câu Nội Bộ vs Dấu Câu Kết Thúc Tập 1](aero_LLM_13_CodeChallenge Clusters in internal vs. terminal punctuation (part 1).md) | [Xem bài viết →](aero_LLM_13_CodeChallenge Clusters in internal vs. terminal punctuation (part 1).md) |
| [Phân Khảo Cấu Trúc Cụm (Clusters): Dấu Câu Nội Bộ vs Dấu Câu Kết Thúc Tập 2](aero_LLM_14_CodeChallenge Clusters in internal vs. terminal punctuation (part 2).md) | [Xem bài viết →](aero_LLM_14_CodeChallenge Clusters in internal vs. terminal punctuation (part 2).md) |
| [Thấu Kính Logit (The Logit Lens): Soi Sáng Tư Duy Tầng Trung Gian Của Mô Hình Ngôn Ngữ](aero_LLM_15_The Logit Lens.md) | [Xem bài viết →](aero_LLM_15_The Logit Lens.md) |
| [Thử Thách Lập Trình (Code Challenge): Ứng Dụng Logit Lens Trong Mạng BERT (Phần 1)](aero_LLM_16_CodeChallenge Logit Lens in BERT (part 1).md) | [Xem bài viết →](aero_LLM_16_CodeChallenge Logit Lens in BERT (part 1).md) |
| [Thử Thách Lập Trình (Code Challenge): Ứng Dụng Logit Lens Trong Mạng BERT (Phần 2)](aero_LLM_17_CodeChallenge Logit Lens in BERT (part 2).md) | [Xem bài viết →](aero_LLM_17_CodeChallenge Logit Lens in BERT (part 2).md) |
| [Phân Tích Sự Tương Đồng Tokens Trong và Giữa Các Ma Trận Q, K, V (Phần 1)](article_aero_LLM_01_vn.md) | [Xem bài viết →](article_aero_LLM_01_vn.md) |
| [Phân tích Chuyên Sâu Các Tầng Ẩn Trong Mô Hình Ngôn Ngữ Lớn (LLMs): Đo Lường, Biểu Diễn và Giải Mã Nội Tại](scientific_article_vn.md) | [Xem bài viết →](scientific_article_vn.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
