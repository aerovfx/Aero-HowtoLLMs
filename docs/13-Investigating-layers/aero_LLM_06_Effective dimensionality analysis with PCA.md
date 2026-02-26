
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
# Phân Tích Số Chiều Hiệu Quả (Effective Dimensionality) Thông Qua PCA

## Tóm tắt (Abstract)
Kỹ thuật Phân Tích Thành Phần Chính (PCA) kết hợp với Phân Rã Giá Trị Đặc Dị (SVD) được ứng dụng để định lượng "Số Chiều Hiệu Quả" (Effective Dimensionality) của các ma trận biểu diễn trong LLMs. Nghiên cứu thực hiện trên GPT-2 XL, đối chứng chuỗi token thực tế và chuỗi token bị xáo trộn ngẫu nhiên (shuffled tokens) xuyên suốt khu vực $Hidden\ States$ từ Tầng Embeddings lên 48 Transformer Blocks. Kết quả xây dựng một đồ thị biến thiên không gian nhiều chiều (laminar profile) thể hiện khả năng co giãn (expansion and contraction) tính chất dữ liệu khi mô hình bắt đầu tổ hợp ngữ cảnh ngôn ngữ.

---

## 1. Mở Đầu (Introduction)
Dữ liệu nhúng của một token có thể tồn tại trong một ma trận rất rộng, ví dụ GPT-2 XL có số chiều không gian xung quanh (ambient dimensionality) lên tới 1600. Tuy nhiên, thông tin thực sự hữu ích lại thường rải rác chạy trên một đa tạp (manifold) ít chiều hơn rất nhiều. Tính chất này được gọi là **Số Chiều Hiệu Quả (Effective Dimensionality)**. Đo lường chính xác các biên độ không gian này cung cấp lợi thế phân tích lượng thông tin mang tính tập trung cao, đồng thời theo dõi sát được năng lực nhồi nén và thu phóng ngữ cảnh ở từng chạm (layers) của Large Language Models.

---

## 2. Phương Pháp Toán Học (Mathematical Methodology)

### 2.1. Phân Rã SVD Và Khai Vấn PCA
Ma trận kích hoạt đã được chuẩn hóa trung tâm (Mean-Centering) $X$ được phân rã $SVD$:
$$ X = U \Sigma V^T $$
Với mục tiêu tìm kiếm mức phân tán dữ liệu, PCA gạt bỏ ma trận vector Unit ( $U, V$ ) và chỉ sử dụng **Singular Values** $(\sigma_i)$ trên đường chéo $\Sigma$ làm trọng số tính toán phương sai. Do 100% biến thiên dữ liệu nằm ngọn ở đây, Phần trăm phương sai được giải thích (Percent Variance Explained) của một thành phần (component) thứ $i$ được lập công thức:
$$ r^2_i = \left( \frac{\sigma_i^2}{\sum_{j} \sigma_j^2} \right) \times 100\% $$

### 2.2. Đo Lường Mốc Số Chiều Hiệu Quả
Thay vì giữ số chiều chết 1600, ta tính luỹ kế phần trăm phương sai (Cumulative Sum) và đặt một ngưỡng cắt lọc nhiễu (Ví dụ 95%). "Số Chiều Hiệu Quả" chính thức là số hiệu component nhỏ nhất sao cho rào cản luỹ kế $\ge 95\%$ vừa bị vượt qua. Các chiều không gian dư thặng đằng sau có nồng độ thông tin quá nhỏ, sẽ bị xem là vi tạp (Noise).

---

## 3. Thực Nghiệm Động Chuyển Token (Tokens Shuffling Setup)

Để chứng minh chiều không gian nở ra nhờ có "cú pháp" câu từ, ta sử dụng 1000 tokens văn bản thực (Trích xuất từ sách *Through the Looking Glass*). Tương phản đối diện là khối 1000 tokens đó nhưng bị đảo tung vị trí (Shuffled Token sequences).

Thiết lập đo số chiều tại cả 49 trạm (1 lớp embeddings khởi điểm + 48 vòng lặp Transformer Blocks):
Ngưỡng Effective Dimensionality được hệ thống dò tìm tự động tương ứng $95\%$ Variances.

---

## 4. Khám Phá Sự Co Giãn Đa Không Gian (Analysis & Results)

Laminar Profile hiện thực hóa các đường cong biến động (Dimensionality Expansion and Contraction):

1. **Khởi điểm - Tầng Embeddings (Layer 0):** Dữ liệu hoàn toàn chưa đi qua ma trận quy nạp $Attention$, cả câu từ chuẩn gốc hay câu xáo trộn đều chỉ là từ ghép cơ học. Biến động không gian xuất phát (effective dimensionality) của hai dãy dữ liệu bằng y hệt nhau.
2. **Khu vực phình nở (Expansion) - Tầng Nông:** Lên các block tiếp theo, khi $Q, K, V$ làm việc hút ngữ cảnh cấu trúc đứng trước, dãy Token chuẩn gốc ép mạng không gian phải mở rộng nhanh chóng nhằm dung nạp đa liên kết ngữ pháp. 
3. **Khu vực giới hạn (Contraction) - Đi sâu hệ thống:** Dữ liệu dần bị thắt cổ chai, gom nén lại vào những đặc trưng ngữ nghĩa mang định hướng dự đoán từ tiếp theo. Số chiều thay vì nở to, bắt đầu thu hẹp cục bộ dần xuống.
4. **Đối chiếu văn bản mổ cò (Shuffled Differences):** Đáng ngạc nhiên, luồng token xáo trộn bị mô hình định giá là thông tin rối loạn ngôn ngữ. Do bản chất rời rạc, không chứa mấu chốt cấu trúc văn phạm có thể dự đoán, số chiều được model huy động ít hơn rất rõ rệt và quỹ đạo co giãn không mượt mà như dãy dữ liệu sạch.

---

## 5. Kết Luận (Conclusion)
Phân tích giới hạn số chiều lưu trữ (Dimensionality PCA) hé lộ mô hình tự điều hướng tài nguyên không gian rất khôn khéo. Sức sống của thuật toán Self-Attention không chỉ là nhặt điểm vector từ một "Túi từ" (Bag-of-words) lớn, mà là một quy trình tái cấu trúc không gian hình học. Các token có hệ thống văn phạm đòi hỏi một đại dương nhiều chiều hơn để tổ hợp biểu diễn hơn là các mẩu từ vựng vô nghĩa đứng rời rạc.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh thí nghiệm liên kết: `aero_LLM_06_Effective dimensionality analysis with PCA.md` (Giới thiệu các hàm tính PCA, r-squared variances, SVD values và hiện tượng mở rộng/thu hẹp số chiều trên không gian Hidden States của LLM).
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
