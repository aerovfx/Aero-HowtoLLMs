
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
# Phân Tích Đối Chiếu Đo Lường Tương Quan: Mutual Information và Covariance

## Tóm tắt (Abstract)
Báo cáo này tập hợp và so sánh trực tiếp hai giải pháp toán học đo lường sự phụ thuộc thống kê (statistical dependencies) phổ biến đối với tín hiệu lưới (Network Activations): **Thông Tin Tương Hỗ (Mutual Information - MI)** và **Hiệp Phương Sai (Covariance)**. Thông qua kiểm thử dữ liệu mô phỏng tuyến tính và phi tuyến tính (simulated linear/non-linear data), nghiên cứu phơi bày ưu/nhược điểm cốt lõi của từng phương pháp. Đặc biệt, phân tích làm rõ bản chất "không dấu" (unsigned) của MI so với tính chất phân cực (signed) của Covariance, cũng như tốc độ tính toán và độ nhạy cảm của chúng đối với không gian đo kiểm.

---

## 1. Mở Đầu (Introduction)
Covariance và Mutual Information đều nhằm trả lời một câu hỏi cơ bản: *"Khi tôi biết hoạt động của Tín hiệu X, tôi có thể dự đoán được phần nào hành vi của Tín hiệu Y hay không?"*
Dù có chung mục đích, cách chúng đánh giá dữ liệu lại nằm ở hai hệ quy chiếu khác biệt. Việc hiểu rõ ranh giới toán học của hai công cụ này đóng vai trò quan trọng trước khi áp dụng chúng để bóc tách hành vi của Mạng Neural.

---

## 2. Lý Thuyết Phương Pháp Cốt Lõi (Core Methodologies)

### 2.1. Hiệp Phương Sai (Covariance)
Covariance là một đo lường "tuyến tính" thuần tuý và được lấy trực tiếp trên giá trị tuyệt đối của dữ liệu.
Đối với 2 biến trung tâm hóa (mean-centered) X và Y:
$$ Cov(X,Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y}) $$
**Ưu điểm:**
- Nhanh, mạnh và cực kỳ ổn định về mặt số học.
- Bảo tồn tỷ lệ (scale) của dữ liệu (Ví dụ: dữ liệu đơn vị "mét" thì covariance đơn vị "mét vuông"). Tính chất này đặc biệt hữu dụng với các bài toán truy vết biên độ (Magnitude Tracking).
- Định dạng có dấu (Signed): Nó báo cho bạn biết X và Y là đi lên cùng nhau (Dấu +) hay nghịch biến (Dấu -). 

### 2.2. Thông Tin Tương Hỗ (Mutual Information - MI)
MI không lấy theo số liệu gốc mà phân rã dữ liệu vào ma trận Histogram trước, sau đó tính toán trên không gian xác suất (probability distribution).
$$ I(X;Y) = \sum_{x} \sum_{y} P(x,y) \log \left( \frac{P(x,y)}{P(x)P(y)} \right) $$
**Ưu điểm:**
- Đoán nhận được cả cấu trúc tương quan tuyến tính lẫn phi tuyến tính (đường cong).
- Giải phóng khỏi rào cản tỷ lệ (scale-independence). MI của 1 triệu hay 1 tỷ cũng không làm thay đổi giá trị thông tin nền tảng.

---

## 3. Khám Phá Qua Dữ Liệu Mô Phỏng (Analysis & Results)

### 3.1. Phản Ứng Với Dữ Liệu Phi Tuyến (Non-linear Dependency)
Khi sinh một mảng dữ liệu có mô hình parabol (Dạng phễu) hoặc hàm sóng (Cosine) thì kết quả bộc lộ cực kỳ rõ ràng ranh giới của 2 kỹ thuật:
- **Covariance** mù lòa trước các đường cong gập và bị triệt tiêu, và trả về kêt quả xấp xỉ $0$.
- **Mutual Information** ngay lập tức nhận ra trật tự ẩn này và sinh ra lượng thông tin tích cực cao $> 0$.

### 3.2. Tính Phân Cực Và Tính Không Dấu (Signed vs Unsigned metrics)
Đối với biến mô phỏng tuyến tính, khi quét $Covariance$ theo một dải biến thiên đảo cực từ $+0.9$ (Thuận nghịch) xuống thẳng $-0.9$ (Trái nghịch):
- Biểu đồ $Covariance$ tuân thủ mô hình đường thẳng tịnh tiến âm dương hoàn hảo.
- Biểu đồ $MI$ bẻ phễu thành chữ U. Lý do là MI là chỉ số **Unsigned (không dấu)**. Nó chỉ quan tâm đến sức mạnh của thông tin dự đoán. Dù đường truyền nghịch biến (-0.9) hay đồng biến (0.9), "mức độ gợi ý thông tin cho MI" là như nhau và đều $\to \text{Max}$. Để làm cho Covariance có đồ hình tương quan như MI, chỉ cần bình phương Covariance (Squared Covariance).

---

## 4. Kết Luận
Covariance và Mutual Information là hai con dao phân tích của Data Science, mỗi loại sở hữu một công năng chế tác riêng biệt:
- Nếu mạng LLM đang được đánh giá là một khối phân phối giả định (normally distributed data) chứa các tính toán ma trận ma sát tuyến tính đơn thuần $\to$ **Covariance** là giải pháp tối ưu vì sự bền bỉ, dễ diễn dịch âm/dương, và tính toán tức thời.
- Nếu bạn cần nhặt ra các bí mật ẩn giấu dưới dạng quan hệ cấu trúc phức tạp, bất chấp khoảng đo đạc vô cùng hẹp hoặc nhiễu loạn $\to$ **Mutual Information** sẽ là chiếc la bàn đo lường độ bất định thông tin sâu sắc. Tuy nhiên cần hết sức cảnh giác với sai số do hiện tượng lấy mẫu dưới số lượng (Undersampling probabilities).

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh so sánh tĩnh: `aero_LLM_10_Mutual information vs. covariance.md` (Hướng dẫn lập các biến x và y2 (hàm mũ hai, hàm sóng cos) nhằm hiển thị đồ thị chữ U so gánh Covariance signed và MI unsigned metrics).
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
| [Lý Thuyết Thông Tin: Đo Lường Entropy Và Mutual Information](aero_LLM_08_Mutual information theory and code.md) | [Xem bài viết →](aero_LLM_08_Mutual information theory and code.md) |
| [Phân Tích Thông Tin Tương Hỗ Dọc Theo Các Tầng Của Mô Hình Ngôn Ngữ (Pairwise Mutual Information Through LLMs)](aero_LLM_09_Pairwise mutual information through the LLM.md) | [Xem bài viết →](aero_LLM_09_Pairwise mutual information through the LLM.md) |
| 📌 **[Phân Tích Đối Chiếu Đo Lường Tương Quan: Mutual Information và Covariance](aero_LLM_10_Mutual information vs. covariance.md)** | [Xem bài viết →](aero_LLM_10_Mutual information vs. covariance.md) |
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
