
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [12 Investigating neurons dimensions](../index.md)

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
# Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)

## Tóm tắt (Abstract)
Báo cáo này trình bày kết quả của một thực nghiệm khoa học quan trọng: Kiểm chứng tính lặp lại (Reproducibility) của phương pháp Cực đại hóa Hoạt hóa trên GPT-2 Small. Bằng cách lặp lại quy trình tối ưu hóa gradient 10 lần với các điểm khởi đầu ngẫu nhiên khác nhau, nghiên cứu tìm cách xác định liệu mô hình có hội tụ về một "chuỗi token lý tưởng" duy nhất cho một chiều nơ-ron cụ thể hay không. Kết quả cho thấy tính lặp lại cực kỳ thấp (48/50 tokens mang tính duy nhất), cung cấp bằng chứng thực nghiệm về tính chất "hỗn loạn" của không gian biểu diễn và thách thức trong việc xác định các đặc điểm ngôn ngữ ổn định thông qua tối ưu hóa ngược.

---

## 1. Mở Đầu (Introduction)
Trong khoa học, một phát hiện chỉ được coi là có giá trị nếu nó có thể lặp lại được. Nếu quá trình Cực đại hóa Hoạt hóa thực thực sự tìm thấy một "khái niệm" mà nơ-ron đại diện, thì bất kể ta bắt đầu từ nhiễu ngẫu nhiên nào, thuật toán hội tụ nên dẫn tới cùng một kết quả (hoặc ít nhất là các kết quả tương đồng về mặt ngữ nghĩa). Thử thách này thiết lập một quy trình đo lường định lượng cho sự hội tụ này.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Đóng gói Quy trình Tối ưu hóa (Exercise 1)
Xây dựng hàm `train_optimization` để tự động hóa:
1. Khởi tạo ma trận nhúng ngẫu nhiên (5 tokens x 768 dims).
2. Chạy Adam Optimizer qua 500 epochs.
3. Cực đại hóa hoạt hóa trung bình của layer 8, chiều 91.
4. Trả về vector nhúng đã tối ưu.

### 2.2. Kiểm chứng Tính lặp lại (Exercise 2)
- **Thiết lập:** Lặp lại hàm trên 10 lần độc lập.
- **Giải mã:** Chuyển đổi 50 vectors kết quả (10 runs x 5 tokens) thành các tokens thực tế dựa trên độ tương quan Cosine cực đại.
- **Định lượng:** Sử dụng `numpy.unique` để đếm số lượng token duy nhất và tần suất xuất hiện của chúng.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Sự Phân tán của Kết quả (Null Results)
Thực nghiệm cho thấy trong số 50 tokens thu được, có tới 48 tokens là duy nhất. Chỉ có 2 trường hợp lặp lại (ví dụ: token "sup"). 
- **Diễn giải:** Tỷ lệ lặp lại (2/50) là vô cùng nhỏ, điều này chứng tỏ nơ-ron đích không có một "ngôn ngữ mẹ đẻ" cố định mà chúng ta có thể dễ dàng giải mã được bằng phương pháp này.

### 3.2. Hiệu ứng của Điểm khởi đầu (Initialization Bias)
Mặc dù mọi thông số huấn luyện (optimizer, loss function, model weights) là cố định, sự khác biệt duy nhất là nhiễu khởi tạo. Việc kết quả bị phân tán mạnh mẽ chỉ ra rằng nơ-ron đang phản ứng với các cấu trúc toán học trừu tượng trong embedding không gian – những cấu trúc này có thể được thỏa mãn bởi nhiều tổ hợp token khác nhau một cách ngẫu nhiên.

---

## 4. Thảo Luận: Giá trị của các "Phát hiện Âm tính" (Null Findings)
Báo cáo khẳng định rằng kết quả "không lặp lại" vẫn mang giá trị tri thức cao:
1. **Tính Phức tạp:** Nó xác nhận rằng LLM không hoạt động dựa trên các "nhãn từ điển" đơn giản.
2. **Yêu cầu về Ràng buộc:** Để phương pháp này hiệu quả, cần bổ sung các ràng buộc (priors) như tính trơn tru của văn bản hoặc nén chiều không gian, thay vì tối ưu hóa hoàn toàn ngẫu nhiên.
3. **Độ hạt (Granularity):** Kết quả có thể khả quan hơn nếu chúng ta tập trung vào các nơ-ron MLP chuyên biệt thay vì các chiều trong residual stream tổng quát.

---

## 5. Kết Luận
Thử thách này minh chứng rằng Cực đại hóa Hoạt hóa nguyên bản là một công cụ không ổn định cho việc diễn giải ngôn ngữ. Sự thiếu hụt tính lặp lại mở ra nhu cầu cho các kỹ thuật trích xuất hoạt hóa tinh vi hơn (như Hooks trực tiếp vào nội bộ Transformer Block) và các phương pháp thống kê thay thế như lấy mẫu dữ liệu diện rộng.

---

## Tài liệu tham khảo (Citations)
1. Thử thách về tính lặp lại của Activation Maximization trên GPT-2 Small dựa trên `aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md`. Phân tích sự phân tán của 10 lần chạy độc lập.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12-Investigating-neurons-dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_LLM_01_Activation maximization via gradient ascent (theory).md) | [Xem bài viết →](aero_LLM_01_Activation maximization via gradient ascent (theory).md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_LLM_02_Activation maximization (code).md) | [Xem bài viết →](aero_LLM_02_Activation maximization (code).md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_LLM_03_Activation maximization via data sampling.md) | [Xem bài viết →](aero_LLM_03_Activation maximization via data sampling.md) |
| 📌 **[Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md)** | [Xem bài viết →](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) |
| [Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)](aero_LLM_05_Extracting activations using hooks.md) | [Xem bài viết →](aero_LLM_05_Extracting activations using hooks.md) |
| [Mối tương quan giữa Hooks và Hidden States: Giải cấu trúc Khối Transformer (Reconstructing Transformer Blocks)](aero_LLM_06_Relation between hooks and output.hidden_states.md) | [Xem bài viết →](aero_LLM_06_Relation between hooks and output.hidden_states.md) |
| [Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)](aero_LLM_07_Clarification of final hidden_states output.md) | [Xem bài viết →](aero_LLM_07_Clarification of final hidden_states output.md) |
| [Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 1)](aero_LLM_08_CodeChallenge Grammar tuning in MLP neurons (part 1).md) | [Xem bài viết →](aero_LLM_08_CodeChallenge Grammar tuning in MLP neurons (part 1).md) |
| [Thử thách Lập trình: Tính Chọn lọc Ngữ pháp của Nơ-ron MLP (Phần 2)](aero_LLM_09_CodeChallenge Grammar tuning in MLP neurons (part 2).md) | [Xem bài viết →](aero_LLM_09_CodeChallenge Grammar tuning in MLP neurons (part 2).md) |
| [Thử thách Lập trình: Sự Điều chế Ngữ cảnh trong Hoạt hóa MLP (Context-modulated Activation)](aero_LLM_10_CodeChallenge Context-modulated activation in MLP.md) | [Xem bài viết →](aero_LLM_10_CodeChallenge Context-modulated activation in MLP.md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 1)](aero_LLM_11_CodeChallenge Activation histograms by token length (part 1).md) | [Xem bài viết →](aero_LLM_11_CodeChallenge Activation histograms by token length (part 1).md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 2)](aero_LLM_12_CodeChallenge Activation histograms by token length (part 2).md) | [Xem bài viết →](aero_LLM_12_CodeChallenge Activation histograms by token length (part 2).md) |
| [Thử thách Lập trình: Độ dài Token và Đặc tính Hoạt hóa (Phần 3)](aero_LLM_13_CodeChallenge Activation histograms by token length (part 3).md) | [Xem bài viết →](aero_LLM_13_CodeChallenge Activation histograms by token length (part 3).md) |
| [Xử lý Biểu diễn Nơ-ron cho các Từ đa Token (Multi-token Words)](aero_LLM_14_Dealing with multitoken word embeddings.md) | [Xem bài viết →](aero_LLM_14_Dealing with multitoken word embeddings.md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 1)](aero_LLM_15_CodeChallenge Category-tuned MLP projections (part 1).md) | [Xem bài viết →](aero_LLM_15_CodeChallenge Category-tuned MLP projections (part 1).md) |
| [Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 2)](aero_LLM_16_CodeChallenge Category-tuned MLP projections (part 2).md) | [Xem bài viết →](aero_LLM_16_CodeChallenge Category-tuned MLP projections (part 2).md) |
| [Hồi quy Logistic: Lý thuyết và Triển khai Phân loại Nơ-ron](aero_LLM_17_Classification via logistic regression theory and code.md) | [Xem bài viết →](aero_LLM_17_Classification via logistic regression theory and code.md) |
| [Đối chiếu Hồi quy Logistic và Kiểm định T-test: Giả định và Ứng dụng](aero_LLM_18_Logistic regression vs. t-test assumptions and applications.md) | [Xem bài viết →](aero_LLM_18_Logistic regression vs. t-test assumptions and applications.md) |
| [Điều chỉnh Danh từ riêng trong GPT-2 Medium](aero_LLM_19_Proper noun tuning in GPT2-medium.md) | [Xem bài viết →](aero_LLM_19_Proper noun tuning in GPT2-medium.md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) | [Xem bài viết →](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) | [Xem bài viết →](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) | [Xem bài viết →](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
