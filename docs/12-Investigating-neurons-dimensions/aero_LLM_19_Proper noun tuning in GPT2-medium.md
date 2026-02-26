
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
# Điều chỉnh Danh từ riêng trong GPT-2 Medium

## Tóm tắt (Abstract)
Báo cáo này sử dụng Hồi quy Logistic để xác định và phân tích các nơ-ron MLP chuyên biệt hóa cho việc nhận diện danh từ riêng (Proper Nouns) trong mô hình GPT-2 Medium. Bằng cách khai thác đặc điểm hình thái của tiếng Anh (viết hoa các danh từ riêng), chúng ta triển khai một thuật toán lọc tự động trên tập dữ liệu WikiText để phân loại token. Nghiên cứu thực nghiệm trên toàn bộ 4096 nơ-ron của một tầng MLP bộc lộ sự tồn tại của các "đơn vị danh từ riêng" (proper noun units) với độ tin cậy thống kê cao. Tính bền vững của phát hiện được kiểm chứng thông qua các bản đồ nhiệt văn bản trên các mẫu dữ liệu chưa từng xuất hiện trong quá trình huấn luyện hồi quy.

---

## 1. Mở Đầu (Introduction)
Danh từ riêng đại diện cho các thực thể cụ thể (người, địa danh, tổ chức). Trong cơ chế nội soi mô hình, việc hiểu cách LLM phân tách danh từ riêng khỏi các danh từ chung và các thành phần ngữ pháp khác là chìa khóa để giải mã cách mô hình xây dựng bản đồ tri thức thế giới. Nghiên cứu này tập trung vào GPT-2 Medium – một mô hình có kích thước trung bình với 24 khối Transformer và 4096 nơ-ron trong mỗi lớp mở rộng MLP.

---

## 2. Xác định Danh từ riêng bằng Thuật toán

### 2.1. Quy tắc Phân loại Hình thái
Do cấu trúc dữ liệu Wikipedia chứa mật độ danh từ riêng cao, chúng ta áp dụng một thuật toán lọc đơn giản nhưng hiệu quả:
1. **Điều kiện Chữ hoa:** Token sau khi loại bỏ khoảng trắng (`strip()`) phải bắt đầu bằng một chữ cái viết hoa.
2. **Loại trừ Đầu câu:** Để tránh nhầm lẫn với các từ được viết hoa do đứng đầu câu, chúng ta kiểm tra token ngay phía trước. Nếu token đó kết thúc bằng dấu chấm (.), dấu chấm hỏi (?) hoặc dấu chấm than (!), token hiện tại sẽ bị loại khỏi nhóm danh từ riêng.

### 2.2. Chuẩn bị Mẫu So sánh
Để hồi quy vận hành tối ưu, chúng ta thiết lập hai nhóm có kích thước bằng nhau ($n \approx 220$):
- **Nhóm Đích (Target):** Các danh từ riêng hợp lệ.
- **Nhóm Đối chứng (Comparison):** Các token khác được chọn ngẫu nhiên từ cùng một batch dữ liệu (bao gồm động từ, giới từ, số, v.v.).

---

## 3. Phân tích Định lượng xuyên Tầng

### 3.1. Phân phối Hệ số Beta
Thực hiện hồi quy trên 4096 nơ-ron và áp dụng hiệu chỉnh Bonferroni ($p < 0.05 / 4096$):
- **Beta Dương ($\beta > 0$):** Chỉ thị các nơ-ron có hoạt hóa mạnh khi gặp danh từ riêng. Đây là các đối tượng nghiên cứu chính.
- **Beta Âm ($\beta < 0$):** Chỉ thị các nơ-ron bị ức chế hoạt hóa khi gặp danh từ riêng. Mặc dù khó diễn giải hơn, hiện tượng này tương đồng với cơ chế ức chế chọn lọc (selective inhibition) thường thấy trong thần kinh học sinh học.

### 3.2. Trực quan hóa Beta vs. P-value
Biểu đồ scatter plot giữa hệ số hồi quy và $-\log(p)$ cho thấy một cấu trúc hình phễu: các nơ-ron có hiệu ứng mạnh nhất ($\beta$ lớn) cũng đồng thời là các nơ-ron có ý nghĩa thống kê cao nhất.

---

## 4. Phân tích Định tính và Kiểm chứng Bền vững

### 4.1. Bản đồ nhiệt Văn bản (Text Heatmap)
Bằng cách trực quan hóa hoạt hóa của nơ-ron có $\beta$ cực đại lên các đoạn văn bản, chúng ta quan sát thấy sự "thắp sáng" rõ rệt tại các tên người và địa danh. 
- **Lưu ý về Polysemanticity:** Một số nơ-ron có thể kích hoạt cả với các thực thể liên quan (ví dụ: kích hoạt với tên người và các từ liên quan đến truyền hình).

### 4.2. Kiểm chứng Ngoài mẫu (Out-of-sample Validation)
Để loại trừ hiện tượng quá khớp (overfitting), nơ-ron được xác định từ batch 1 được kiểm tra trên batch 2. Kết quả duy trì được tính chọn lọc danh từ riêng, khẳng định rằng nơ-ron này đã thực sự học được khái niệm trừu tượng về danh từ riêng thay vì chỉ ghi nhớ các từ cụ thể.

---

## 5. Kết Luận
Nghiên cứu xác nhận rằng GPT-2 Medium sở hữu các kênh xử lý chuyên biệt cho danh từ riêng nằm trong lớp MLP. Việc sử dụng Hồi quy Logistic kết hợp với các thuật toán lọc ngôn ngữ cung cấp một công cụ mạnh mẽ để "bản đồ hóa" chức năng của hàng nghìn nơ-ron trong các mô hình ngôn ngữ lớn.

---

## Tài liệu tham khảo (Citations)
1. Nghiên cứu Proper noun tuning trên GPT-2 Medium dựa trên `aero_LLM_19_Proper noun tuning in GPT2-medium.md`. Phân tích hệ số Beta và kiểm chứng ngoài mẫu bằng Text Heatmap.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 12-Investigating-neurons-dimensions](README.md) | [Xem bài viết →](README.md) |
| [Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM](aero_LLM_01_Activation maximization via gradient ascent (theory).md) | [Xem bài viết →](aero_LLM_01_Activation maximization via gradient ascent (theory).md) |
| [Triển khai Cực đại hóa Hoạt hóa: Từ Gradient Ascent đến Giải mã Token (Activation Maximization Implementation)](aero_LLM_02_Activation maximization (code).md) | [Xem bài viết →](aero_LLM_02_Activation maximization (code).md) |
| [Cực đại hóa Hoạt hóa qua Lấy mẫu Dữ liệu (Activation Maximization via Data Sampling)](aero_LLM_03_Activation maximization via data sampling.md) | [Xem bài viết →](aero_LLM_03_Activation maximization via data sampling.md) |
| [Thử thách Lập trình: Kiểm chứng Tính lặp lại của Cực đại hóa Hoạt hóa (Reproducibility of Activation Maximization)](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) | [Xem bài viết →](aero_LLM_04_CodeChallenge Reproducibility of activation maximization.md) |
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
| 📌 **[Điều chỉnh Danh từ riêng trong GPT-2 Medium](aero_LLM_19_Proper noun tuning in GPT2-medium.md)** | [Xem bài viết →](aero_LLM_19_Proper noun tuning in GPT2-medium.md) |
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
