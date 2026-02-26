
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
# Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)

## Tóm tắt (Abstract)
Báo cáo này hướng dẫn phương pháp sử dụng "Hooks" – các hàm can thiệp đặc biệt trong PyTorch – để truy cập và trích xuất dữ liệu từ các lớp ẩn bên trong Transformer. Trong khi các phương thức thông thường chỉ cho phép quan sát Logits đầu ra hoặc Hidden States của toàn bộ khối Transformer, kỹ thuật Hook cho phép nhà nghiên cứu cô lập các thành phần vi mô như ma trận Query (Q), Key (K), Value (V) hoặc các lớp MLP. Báo cáo cũng thảo luận về cơ chế quản lý Hook (đăng ký và gỡ bỏ) và cách quản lý bộ nhớ thông qua việc ghi đè hoặc tích lũy dữ liệu.

---

## 1. Mở Đầu (Introduction)
Để thực hiện Diễn giải học cơ học (Mechanistic Interpretability), việc biết trọng số (weights) của mô hình là chưa đủ. Chúng ta cần biết cách các nơ-ron thực sự phản ứng (activations) khi dữ liệu cụ thể đi qua. Hooks đóng vai trò như các "cảm biến" được cấy vào dòng chảy dữ liệu của mô hình trong quá trình forward-pass, cho phép ta chụp lại trạng thái của bất kỳ nơ-ron nào mà không cần sửa đổi cấu trúc cốt lõi của mạng.

---

## 2. Cơ chế Hoạt động của PyTorch Hooks

### 2.1. Định nghĩa Hàm Hook
Một hàm Hook tiêu chuẩn nhận ba tham số đầu vào:
1. **Module:** Lớp (layer) mà hook được gắn vào.
2. **Input:** Dữ liệu đi vào lớp đó.
3. **Output:** Kết quả tính toán đi ra khỏi lớp đó.
Bên trong hàm này, ta có thể trích xuất `output`, thực hiện các phép toán (như tách các chiều Q, K, V) và lưu trữ kết quả vào một biến bên ngoài (thường là Dictionary hoặc List).

### 2.2. Đăng ký và Quản lý (Registration & Handles)
Sử dụng phương thức `register_forward_hook` để cấy hàm vào mô hình. Kết quả trả về là một `handle`, có thể được sử dụng để gỡ bỏ (`remove()`) hook khi không còn cần thiết, giúp tối ưu hóa hiệu năng và tránh rò rỉ bộ nhớ.

---

## 3. Quản lý Dữ liệu Hoạt hóa (Data Management)

### 3.1. Ghi đè (Overwriting via Dictionary)
Nếu lưu trữ dữ liệu vào một `Dictionary` với key là tên tầng, mỗi lượt forward-pass mới sẽ ghi đè lên dữ liệu cũ. Đây là cách tiếp cận phổ biến khi ta chỉ quan tâm đến phản hồi của mô hình đối với câu lệnh hiện tại. 
*Lưu ý:* Nếu câu lệnh mới có các token đầu tiên giống câu lệnh cũ, các hàng tương ứng trong ma trận hoạt hóa sẽ giống nhau do tính chất truyền tin theo trình tự.

### 3.2. Tích lũy (Accumulation via List)
Bằng cách sử dụng `List` và phương thức `append()`, ta có thể lưu trữ lịch sử hoạt hóa của tất cả các câu lệnh đã đi qua mô hình. Điều này hữu ích cho các phân tích thống kê diện rộng hoặc so sánh sự biến thiên của nơ-ron qua nhiều ngữ cảnh khác nhau.

---

## 4. Phân tích Dữ liệu trích xuất
Khi đã có dữ liệu qua Hook, ta có thể thực hiện các phân tích trực quan:
- **Scatter Plots:** So sánh hoạt hóa của hai token khác nhau trên toàn bộ các nơ-ron của một tầng.
- **Correlation Matrices:** Đo lường sự tương quan giữa các token. Quan sát thực nghiệm cho thấy token đầu tiên thường có độ tương quan thấp với phần còn lại do thiếu hụt ngữ cảnh tiền đề.

---

## 5. Kết Luận
Hooks là công cụ mạnh mẽ nhất để biến một mô hình "hộp đen" thành một hệ thống có thể quan sát được ở mọi cấp độ hạt. Việc làm chủ kỹ thuật này không chỉ giúp trích xuất dữ liệu mà còn đặt nền móng cho việc chỉnh sửa hoạt hóa (activation editing) – một kỹ thuật can thiệp nhân quả sâu sắc hơn sẽ được thảo luận ở các chương sau.

---

## Tài liệu tham khảo (Citations)
1. Kỹ thuật trích xuất hoạt hóa bằng Hooks trên GPT-2 dựa trên `aero_LLM_05_Extracting activations using hooks.md`. Phân tích sự khác biệt giữa cơ chế Overwriting và Concatenation.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12-Investigating-neurons-dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_LLM_01_Activation maximization via gradient ascent (theory).md) | [Xem bài viết →](aero_LLM_01_Activation maximization via gradient ascent (theory).md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_LLM_02_Activation maximization (code).md) | [Xem bài viết →](aero_LLM_02_Activation maximization (code).md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_LLM_03_Activation maximization via data sampling.md) | [Xem bài viết →](aero_LLM_03_Activation maximization via data sampling.md) |
| [Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) |
| 📌 **[Giải phẫu Nội tại Mô hình bằng Hooks: Kỹ thuật Trích xuất Hoạt hóa (Extracting Activations via Hooks)](aero_LLM_05_Extracting activations using hooks.md)** | [Xem bài viết →](aero_LLM_05_Extracting activations using hooks.md) |
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
