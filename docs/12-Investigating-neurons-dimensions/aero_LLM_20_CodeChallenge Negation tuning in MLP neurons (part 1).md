
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
# Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này khởi đầu một thử thách lập trình chuyên sâu nhằm xác định các nơ-ron MLP chuyên biệt hóa cho các khái niệm logic nhị phân: Phủ định (Negation) và Khẳng định (Affirmation). Sử dụng mô hình GPT-2 Large và văn bản từ tác phẩm của Philip K. Dick, chúng ta triển khai một quy trình trích xuất hoạt hóa tối ưu thông qua việc cải tiến kỹ thuật cấy Hook. Quy trình bao gồm việc chuẩn bị dữ liệu văn bản thực tế, xây dựng cửa sổ ngữ cảnh (context window) đồng nhất và quản lý bộ nhớ hoạt hóa xuyên suốt các lượt chạy forward pass.

---

## 1. Tối ưu hóa Kỹ thuật Hooks (Exercise 1)

### 1.1. Chuyển đổi từ Input-centric sang Output-centric
Trong các bài thực hành trước, chúng ta thường hook vào lớp MLP tổng thể và thực hiện phép nhân ma trận thủ công bên trong hàm hook để lấy hoạt hóa lớp mở rộng. Tuy nhiên, phương pháp này gây lãng phí tài nguyên do tính toán trùng lặp.
- **Cải tiến:** Hook trực tiếp vào thành phần `c_fc` (lớp nơ-ron mở rộng).
- **Lợi ích:** Lấy trực tiếp giá trị `output` của lớp này, giúp mã nguồn gọn nhẹ và tối ưu hóa tốc độ xử lý khi làm việc với mô hình lớn như GPT-2 Large (5120 nơ-ron mỗi lớp MLP).

---

## 2. Chuẩn bị Dữ liệu Ngôn ngữ (Exercise 2)

### 2.1. Nguồn dữ liệu và Phân loại
Nghiên cứu sử dụng văn bản từ Dự án Gutenberg, cụ thể là các tác phẩm khoa học viễn tưởng để có ngôn ngữ phong phú:
- **Nhóm Phủ định (Negations):** *not, cannot, can't, don't, won't, never, wasn't...*
- **Nhóm Khẳng định (Affirmations):** *can, could, may, will...*

### 2.2. Thuật toán Lọc Token tinh vi
Việc tìm kiếm token đích không chỉ đơn giản là khớp chuỗi ký tự mà cần xử lý các trường hợp chồng lấn (ví dụ: không lấy từ "can" nếu nó là một phần của "cannot").
- **Kiểm tra Token kế tiếp:** Một token được coi là từ đích độc lập chỉ khi token ngay sau nó bắt đầu bằng một khoảng trắng. Điều này giúp loại bỏ các trường hợp token đích là tiền tố của một từ dài hơn (như "connotative").

### 2.3. Cửa sổ Ngữ cảnh Đồng nhất
Để đảm bảo các nơ-ron có đủ thông tin ngữ nghĩa, mỗi từ đích phải nằm trong một cửa sổ: **[90 tokens trước] + [Target Word] + [10 tokens sau]**. Những từ đích xuất hiện quá gần đầu hoặc cuối văn bản sẽ bị loại bỏ để tránh lỗi biên.

---

## 3. Trích xuất Hoạt hóa (Exercise 3)

### 3.1. Cấu trúc Batch
Dữ liệu được tổ chức thành hai tensors Batch có kích thước $[N, 101]$, trong đó 101 là tổng độ dài chuỗi context. Vị trí index 90 trong mọi chuỗi luôn là token đích, giúp đơn giản hóa việc lập chỉ mục (indexing) sau này.

### 3.2. Quản lý Tài nguyên và Ghi đè
Do cơ chế của Hook Dictionary là ghi đè (overwriting) dữ liệu sau mỗi lần gọi `model()`, quy trình thực hiện như sau:
1. Chạy Forward Pass cho nhóm Negation $\rightarrow$ Sao chép dữ liệu từ dictionary sang một biến lưu trữ riêng (`activs_neg`).
2. Chạy Forward Pass cho nhóm Affirmation $\rightarrow$ Sao chép sang `activs_aff`.
Sử dụng `torch.no_grad()` và `model.eval()` là bắt buộc để giải phóng bộ nhớ và vô hiệu hóa các lớp Dropout/BatchNormalization.

---

## 4. Kết Luận Phần 1
Chúng ta đã thiết lập xong "hạ tầng" dữ liệu và hoạt hóa. Đây là nền tảng vững chắc để triển khai các phân tích thống kê sâu hơn (như Hồi quy Logistic) nhằm tìm ra những nơ-ron logic chịu trách nhiệm xử lý sự phủ định trong phần tiếp theo.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Negation tuning trên GPT-2 Large dựa trên `aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md`. Thiết lập Batch context và tối ưu hóa Hook vào lớp `c_fc`.
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
| [Điều chỉnh Danh từ riêng trong GPT-2 Medium](aero_LLM_19_Proper noun tuning in GPT2-medium.md) | [Xem bài viết →](aero_LLM_19_Proper noun tuning in GPT2-medium.md) |
| 📌 **[Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md)** | [Xem bài viết →](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) | [Xem bài viết →](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) | [Xem bài viết →](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
