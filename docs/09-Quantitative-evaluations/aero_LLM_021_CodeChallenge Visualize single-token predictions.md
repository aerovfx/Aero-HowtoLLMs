
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [09 Quantitative evaluations](../index.md)

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
# Thử Thách Lập Trình: Trực Quan Hóa Dự Đoán Đơn Token

## Tóm tắt

Bài viết hướng dẫn thu thập logit, áp dụng hàm `LogSoftmax` và xây dựng bản đồ nhiệt (Text Heatmaps) nhằm hiểu rõ năng lực dự đoán chuỗi ký tự của hai biến thể Mô hình Ngôn ngữ là GPT-2 Small và Large. Điều này đưa ra đánh giá định lượng kèm trực quan nhận thức định tính, giúp các kỹ sư Machine Learning nắm bắt bản chất của sự sinh văn tự.

---

## 1. Truy Xuất Đồ Thị Dữ Liệu Các Token 

Đại lượng cốt lõi $Z_k$ được khai thác ở đây là Logit trả ra từ trạng thái khối transformer cuối cùng của mô hình cho mọi biến token. Trong một chuỗi câu gồm $N$ ký tự $t_i$:

$$[t_1, t_2, \dots, t_{N}]$$

Mô hình sẽ dự đoán:

$$ P(t_{i} | t_1\dots t_{i-1}) $$

### 1.1 Tính Toán Softmax Xác Suất Của Ký Tự Ngay Trước Đó
Bởi ngôn ngữ (Language Model) luôn đóng vai trò phỏng đoán Token cho bước $(t_i)$, ta chỉ có thể sử dụng vị trí $P(t_{i})$ từ đầu ra (output array) của vị trí $t_{i-1}$ (Previous Token).
Mảng dự báo được đưa qua một hàm:
$$ \text{LogSoftmax}(Z_i) = \log\left(\frac{e^{Z_i}}{\sum e^{Z_k}}\right) = Z_i - \log\left(\sum e^{Z_k}\right) $$
LogSoftmax ổn định số học và mang lại sự tinh vi về khoảng độ, vì các xác suất kề $0$ bị làm nhòe. Giá trị logit càng lớn hơn thì độ tương quan với giá trị số âm (xướng lên 0) càng bé dần. Từ đó, xác suất từ token đúng nhất sẽ được trích xuất (Indexed target word).

---

## 2. So Sánh Mô Hình Định Lượng Của Token (Small vs. Large)

Biểu diễn LogSoftmax tại token $t_i$ lấy từ GPT-2 (Small) so sánh trực tiếp (tương tự đồ thị Unity plot) với GPT-2 (Large). Cả hai dùng $t_1 \dots t_{i-1}$ chuẩn làm đầu vào:
- **Tương Quan Thống Kê**: Tỷ lệ Tương quan ($\rho$) thường thấy giữa giá trị LogSoftmax ở hai kiến trúc đạt ngưỡng mạnh $\approx 0.94$. Ở cấp độ bề ngoài, các logits sau chót trả về sự khác biệt không nhiều.
- **Sàn Biến Động**: GPT-2 Large đôi khi cho mức tự tin thấp hơn hoặc cao hơn tùy thuộc vào ngữ pháp. Khẳng định "Mô hình càng lớn thì độ tự tin dự đoán càng cao mọi lần" là sai lầm, bởi mô hình nhỏ cũng có lúc xác suất LogSoftmax rơi vào mức cao cho cùng token.
- Nếu Input và Output giống nhau, ta cũng không được quy nạp "Cơ chế tính toán nội hàm giống hệt nhau". Cấu trúc biểu diễn thông tin bên trong là bất định.

---

## 3. Trực Quan Hóa Nhiệt Độ Ngôn Ngữ (Text Heatmaps)

Chuỗi văn bản (ví dụ "The goal of a correlation...") sẽ được phủ nền bằng hệ số RGB chuẩn lớn tới bé ($0 \to 1$) dựa trên điểm số logit (LogSoftmax values).

### 3.1 Xử Lý Từ Vựng Lỗi (Đầu Câu)
Từ đầu tiên ("The") sẽ không có khả năng nhận định định lượng (Vì ở đó sự tiên tri tương lai từ quá khứ chưa nổ ra). Điều này khiến logit của chữ đầu trở thành số rỗng (0 trên ma trận), nhưng khi chạy bộ Min-Max Scaling (thu giá trị trong khoảng $[0-1]$), thì 0 lại được làm tròn thành $1$, tạo mảng màu sai sự thật lớn nhất.
Để sửa chữa, vòng lặp thường loại bỏ chỉ số từ vị trí thứ $1$ trở vế không (Skipping the zero-th token).

### 3.2 Giải Nghĩa Nhiệt Bức Xạ Token Đầu Ra

- Những chuỗi cụm từ khó có từ đi kèm mặc định (ví dụ: "The goal of a..."): Ma trận có độ tối cực thấp. Năng lực phỏng đoán rời rạc.
- Ngay khi xuất hiện từ có nghĩa hẹp (ví dụ: "...of a correlation..."): Chỉ có số ít từ khả thi (ví dụ: *analysis*, *coefficient*), mô hình có tính đoán đúng gần như truyệt đối (Ma trận kích màu nóng tới max - darkest background).

Đây là nguyên lý hình thái của Language Modeling Generation. Lời văn càng ít tùy chọn, mô hình sẽ sinh số đo Logit xác suất cực độ, ngược lại thì màu nhạt (sự đa dạng văn bản nảy mầm).

---

## Tài liệu tham khảo

1. **Jawahar, G. et al. (2019).** *What Does BERT Learn about the Structure of Language?* ACL.
2. **Kovaleva, O. et al. (2019).** *Revealing the Dark Secrets of BERT.* EMNLP.
3. **Perez, E. et al. (2022).** *Red Teaming Language Models with Language Models.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn](aero_LLM_016_Black box evals.md) | [Xem bài viết →](aero_LLM_016_Black box evals.md) |
| [Red Teaming: Đội Đỏ và Thử Nghiệm Đối Kháng trong AI Safety](aero_LLM_017_Red-teaming.md) | [Xem bài viết →](aero_LLM_017_Red-teaming.md) |
| [Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ](aero_LLM_018_Accuracy, coherence, and relevance.md) | [Xem bài viết →](aero_LLM_018_Accuracy, coherence, and relevance.md) |
| [Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ](aero_LLM_019_Distributions of hidden-state activations.md) | [Xem bài viết →](aero_LLM_019_Distributions of hidden-state activations.md) |
| [Hứa Hẹn và Thách Thức của Đánh Giá Định Lượng trong Mô Hình Học Máy](aero_LLM_01_Promises and challenges of quantitative evaluations.md) | [Xem bài viết →](aero_LLM_01_Promises and challenges of quantitative evaluations.md) |
| [Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)](aero_LLM_020_Heatmaps of tokens for qualitative inspection.md) | [Xem bài viết →](aero_LLM_020_Heatmaps of tokens for qualitative inspection.md) |
| 📌 **[Thử Thách Lập Trình: Trực Quan Hóa Dự Đoán Đơn Token](aero_LLM_021_CodeChallenge Visualize single-token predictions.md)** | [Xem bài viết →](aero_LLM_021_CodeChallenge Visualize single-token predictions.md) |
| [Các Vấn Đề Số Học trong Logits và Softmax: Phân Tích Toán Học và Giải Pháp Ổn Định](aero_LLM_02_Numerical issues in logits and softmax.md) | [Xem bài viết →](aero_LLM_02_Numerical issues in logits and softmax.md) |
| [Perplexity trong Mô Hình Ngôn Ngữ: Cơ Sở Toán Học, Diễn Giải và Giới Hạn](aero_LLM_03_Perplexity.md) | [Xem bài viết →](aero_LLM_03_Perplexity.md) |
| [aero_LLM_04_CodeChallenge Perplexing perplexities.md](aero_LLM_04_CodeChallenge Perplexing perplexities.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Perplexing perplexities.md) |
| [aero_LLM_05_Masked word prediction accuracy.md](aero_LLM_05_Masked word prediction accuracy.md) | [Xem bài viết →](aero_LLM_05_Masked word prediction accuracy.md) |
| [aero_LLM_06_HellaSwag.md](aero_LLM_06_HellaSwag.md) | [Xem bài viết →](aero_LLM_06_HellaSwag.md) |
| [aero_LLM_07_Import large models using bitsandbytes.md](aero_LLM_07_Import large models using bitsandbytes.md) | [Xem bài viết →](aero_LLM_07_Import large models using bitsandbytes.md) |
| [aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md](aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md) | [Xem bài viết →](aero_LLM_08_CodeChallenge HellaSwag evals in two models (part 1).md) |
| [aero_LLM_09_CodeChallenge HellaSwag evals in two models (part 2).md](aero_LLM_09_CodeChallenge HellaSwag evals in two models (part 2).md) | [Xem bài viết →](aero_LLM_09_CodeChallenge HellaSwag evals in two models (part 2).md) |
| [aero_LLM_10_KL (Kullback-Leibler) divergence.md](aero_LLM_10_KL (Kullback-Leibler) divergence.md) | [Xem bài viết →](aero_LLM_10_KL (Kullback-Leibler) divergence.md) |
| [aero_LLM_11_MAUVE.md](aero_LLM_11_MAUVE.md) | [Xem bài viết →](aero_LLM_11_MAUVE.md) |
| [aero_LLM_12_CodeChallenge Large and small MAUVE explorations.md](aero_LLM_12_CodeChallenge Large and small MAUVE explorations.md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Large and small MAUVE explorations.md) |
| [aero_LLM_13_SuperGLUE and other amalgamations.md](aero_LLM_13_SuperGLUE and other amalgamations.md) | [Xem bài viết →](aero_LLM_13_SuperGLUE and other amalgamations.md) |
| [aero_LLM_14_Assessing bias and fairness.md](aero_LLM_14_Assessing bias and fairness.md) | [Xem bài viết →](aero_LLM_14_Assessing bias and fairness.md) |
| [aero_LLM_15_Non-technical benchmarks.md](aero_LLM_15_Non-technical benchmarks.md) | [Xem bài viết →](aero_LLM_15_Non-technical benchmarks.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
