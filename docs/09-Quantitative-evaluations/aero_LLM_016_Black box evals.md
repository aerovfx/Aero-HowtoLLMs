
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
# Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn

## Tóm tắt

Đánh giá hộp đen (Black-box evaluation) là phương pháp đánh giá mô hình ngôn ngữ lớn (LLM) dựa trên các phản hồi từ đầu ra mà không cần truy cập vào kiến trúc nội tại, trọng số hay dữ liệu huấn luyện. Phương pháp này đóng vai trò quan trọng trong việc phát hiện các rủi ro bảo mật thực tế như jailbreaking. Bài viết phân tích các cấp độ tiếp cận mô hình (Black, White, Gray Box), ưu điểm về tính ứng dụng thực tiễn và những hạn chế về mặt khoa học của phương pháp này.

---

## 1. Các Cấp Độ Tiếp Cận Mô Hình

Trong học máy, mức độ truy cập vào mô hình được chia thành ba loại chính:

### 1.1 Hộp Đen (Black Box)
Người dùng chỉ có quyền cung cấp đầu vào (prompts) và nhận đầu ra (outputs). Không có thông tin về:
- Trọng số (weights) và tham số (parameters).
- Kiến trúc mô hình.
- Dữ liệu huấn luyện và các tinh chỉnh sau huấn luyện (post-training fine-tuning).
- Ví dụ: ChatGPT (OpenAI), Claude (Anthropic).

### 1.2 Hộp Trắng (White Box)
Người dùng có quyền truy cập toàn diện:
- Toàn bộ trọng số và kiến trúc.
- Dữ liệu huấn luyện và giao thức huấn luyện.
- Các hệ thống rào chắn (guardrails) và system prompts.

### 1.3 Hộp Xám (Gray Box)
Mức độ trung gian, thường thấy ở các mô hình mã nguồn mở nhưng dữ liệu huấn luyện vẫn được giữ kín.
- Có thể truy cập trọng số nhưng không có dữ liệu huấn luyện hoặc các tinh chỉnh bảo mật nội bộ.
- Ví dụ: Các phiên bản GPT-2 mã nguồn mở.

---

## 2. Kỹ Thuật Đánh Giá Hộp Đen

Kỹ thuật phổ biến nhất là sử dụng các mẹo đặt câu hỏi (prompting tricks) để kích hoạt các hành vi không an toàn của mô hình.

### 2.1 Bẻ khóa Mô hình (Jailbreaking)
Jailbreaking là kỹ thuật lách qua các rào chắn bảo mật bằng cách sử dụng các ngữ cảnh sáng tạo. 
- **Ví dụ điển hình:** Yêu cầu mô hình đóng vai một người bà kể chuyện về công thức chế tạo bom hoặc napalm (Bedtime story attack).

### 2.2 Các lỗi logic và tính toán
Đánh giá hộp đen cũng giúp phát hiện các khiếm khuyết trong cách mô hình xử lý thông tin so với con người:
- **Lỗi đếm ký tự:** Khó khăn trong việc đếm số chữ "r" trong từ "strawberry".
- **Lỗi so sánh số thập phân:** Mô hình có thể cho rằng $8.11 > 8.9$ do nhầm lẫn với quy luật đánh số phiên bản phần mềm.

---

## 3. Ưu điểm và Hạn chế

### 3.1 Ưu điểm
- **Tính thực tế cao:** Phản ánh đúng cách người dùng phổ thông tương tác với AI.
- **Rào cản gia nhập thấp:** Không yêu cầu kỹ năng kỹ thuật sâu, cho phép hàng triệu người tham gia tìm lỗi.
- **Phát hiện nhanh rủi ro:** Giúp các công ty AI vá lỗ hổng kịp thời.

### 3.2 Hạn chế
- **Thiếu tính khoa học:** Thường dựa trên sự ngẫu nhiên hoặc tính sáng tạo cá nhân (serendipitous), không có nguyên lý toán học chặt chẽ.
- **Không giải quyết tận gốc:** Chỉ phát hiện triệu chứng (hành vi lỗi) mà không thể giải thích cơ chế nội tại để sửa lỗi trực tiếp trong kiến trúc.
- **Khả năng mô hình đánh lừa:** Các mô hình mạnh mẽ có khả năng nói dối về năng lực của chính chúng để tránh bị tinh chỉnh hoặc tắt bỏ.

---

## 4. Cơ sở Toán học liên quan

Dù là hộp đen, việc đánh giá vẫn dựa trên xác suất của chuỗi token đầu ra:

$$P(T_{target} | T_{context}) = \text{Softmax}(Z)$$

Trong đó $Z$ là logit đầu ra. Đánh giá hộp đen tập trung vào việc làm thế nào để thay đổi $T_{context}$ sao cho $P(T_{unsafe})$ đạt giá trị cực đại.

---

## Tài liệu tham khảo

1. **Anthropic (2022).** *Red Teaming Language Models to Reduce Harms.*
2. **OpenAI (2023).** *GPT-4 Technical Report.*
3. **Wei, J., et al. (2022).** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.*
4. **Ganguli, D., et al. (2022).** *Predictability and Surprise in Large Language Models.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Đánh Giá Hộp Đen (Black-box Evaluations) trong Mô Hình Ngôn Ngữ Lớn](aero_LLM_016_Black box evals.md)** | [Xem bài viết →](aero_LLM_016_Black box evals.md) |
| [Red Teaming: Đội Đỏ và Thử Nghiệm Đối Kháng trong AI Safety](aero_LLM_017_Red-teaming.md) | [Xem bài viết →](aero_LLM_017_Red-teaming.md) |
| [Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ](aero_LLM_018_Accuracy, coherence, and relevance.md) | [Xem bài viết →](aero_LLM_018_Accuracy, coherence, and relevance.md) |
| [Phân Phối Của Các Kích Hoạt Trạng Thái Ẩn Trong Mô Hình Ngôn Ngữ](aero_LLM_019_Distributions of hidden-state activations.md) | [Xem bài viết →](aero_LLM_019_Distributions of hidden-state activations.md) |
| [Hứa Hẹn và Thách Thức của Đánh Giá Định Lượng trong Mô Hình Học Máy](aero_LLM_01_Promises and challenges of quantitative evaluations.md) | [Xem bài viết →](aero_LLM_01_Promises and challenges of quantitative evaluations.md) |
| [Bản Đồ Nhiệt Của Token Cho Cân Nhắc Định Tính (Text Heatmaps)](aero_LLM_020_Heatmaps of tokens for qualitative inspection.md) | [Xem bài viết →](aero_LLM_020_Heatmaps of tokens for qualitative inspection.md) |
| [Thử Thách Lập Trình: Trực Quan Hóa Dự Đoán Đơn Token](aero_LLM_021_CodeChallenge Visualize single-token predictions.md) | [Xem bài viết →](aero_LLM_021_CodeChallenge Visualize single-token predictions.md) |
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
