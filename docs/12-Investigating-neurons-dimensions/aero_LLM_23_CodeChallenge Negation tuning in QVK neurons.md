
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
# Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)

## Tóm tắt (Abstract)
Báo cáo này mở rộng nghiên cứu về cơ chế phủ định từ các nơ-ron MLP sang các đơn vị trong lớp Attention (Query, Key, Value - QVK). Sử dụng cùng một bộ dữ liệu từ Philip K. Dick và phương pháp Hồi quy Logistic, chúng ta so sánh hành vi của các thành phần Attention với các phát hiện trước đó về MLP. Nghiên cứu triển khai kỹ thuật tách ma trận (tensor splitting) để phân tích độc lập ba kênh Q, K, V. Kết quả xác nhận tính phân tán (distributed) của biểu diễn logic phủ định trong toàn bộ kiến trúc mô hình, đồng thời củng cố quan sát về sự suy giảm tín hiệu "hiện tại" tại các tầng sâu của mạng Transformer.

---

## 1. Thiết lập Thực nghiệm và Hooking Attention

### 1.1. Chuyển đổi Đối tượng Nghiên cứu
Thay vì tập trung vào lớp mở rộng của MLP (`mlp.c_fc`), nghiên cứu chuyển hướng sang lớp tuyến tính của attention (`attn.c_attn`). Trong kiến trúc GPT-2 của OpenAI, các vector Query, Key và Value được tạo ra đồng thời và nối tiếp nhau trong một ma trận rộng.
- **Kích thước:** Trong GPT-2 Large ($d=1280$), lớp này có kích thước $3 \times 1280 = 3840$ đơn vị.
- **Vị trí trích xuất:** Chúng ta thu thập hoạt hóa **trước** khi chúng được đưa vào phương trình tính toán Attention Score (Pre-attention activations).

### 1.2. Tái sử dụng Tài nguyên Dữ liệu
Toàn bộ quy trình lọc token phủ định (*not, won't*) và khẳng định (*can, may*) từ các bài thực hành trước được giữ nguyên. Điều này đảm bảo tính khách quan khi so sánh hiệu năng giữa khối MLP và khối Attention trên cùng một ngữ cảnh ngôn ngữ.

---

## 2. Kỹ thuật Phân tích Đa kênh (Exercise 2)

### 2.1. Hồi quy Logistic trên Ma trận Hợp nhất
Hồi quy được thực hiện trên toàn bộ 3840 đơn vị QVK cùng một lúc. Điều này giúp tối ưu hóa khối lượng tính toán trước khi đi sâu vào chi tiết từng loại vector.

### 2.2. Kỹ thuật Tách Tensor (Tensor Splitting)
Để trực quan hóa sự khác biệt giữa Query, Key và Value, chúng ta cần tách ma trận kết quả:
- **Thách thức:** Các thư viện như NumPy không hỗ trợ trực tiếp hàm tách theo kích thước linh hoạt như PyTorch.
- **Giải pháp:** Chuyển đổi dữ liệu (bao gồm cả các Masked Arrays) sang `torch.tensor` và sử dụng phương thức `.split(n_embed, dim=1)`. Kỹ thuật này cho phép chúng ta cô lập các chỉ số thống kê ($\beta, p$, Accuracy) cho riêng từng thành phần Q, K, V một cách chính xác.

---

## 3. Kết Quả Quan Sát và Đối chiếu

### 3.1. Sự tương đồng với MLP
Xu hướng xuyên tầng của các đơn vị QVK phản chiếu gần như hoàn hảo những gì đã quan sát ở MLP:
- **Tỷ lệ nơ-ron "nhạy cảm":** Giảm từ ~70% ở các tầng đầu xuống dưới 30% ở các tầng cuối.
- **Độ chính xác:** Giảm từ mức 75% xuống gần mức ngẫu nhiên (50-60%) khi tiến về phía Output layer.

### 3.2. Hiệu năng của các đơn vị Q, K, V
Thực nghiệm cho thấy cả ba loại vector (Q, K, V) đều tham gia vào việc mã hóa sự phủ định, nhưng với các quy mô hiệu ứng khác nhau. Việc độ chính xác giảm mạnh ở các tầng cuối trong Attention sublayer càng củng cố giả thuyết rằng mô hình đang ưu tiên tích hợp thông tin liên ngữ cảnh (inter-token) để dự đoán từ tiếp theo hơn là duy trì đặc tính ngữ pháp của từ hiện tại.

---

## 4. Thảo luận và Kết luận

### 4.1. Bản chất Phân tán của LLM
Một kết luận quan trọng rút ra từ chuỗi thử thách này là: Các đặc tính chức năng (như nhận diện phủ định) được phân bổ một cách định lượng (quantitative) thay vì định tính (qualitative). Không có sự phân chia module tuyệt đối; thay vào đó, thông tin về phủ định "thấm" qua cả MLP và Attention, giảm dần theo chiều sâu nhưng không bao giờ biến mất hoàn toàn.

### 4.2. Bài học về Lập trình Khoa học
Việc dành thời gian để viết và hiểu từng dòng mã nguồn, thay vì phụ thuộc vào các công cụ AI tạo mã, là khoản đầu tư cần thiết để nắm bắt được các sắc thái tinh tế trong dữ liệu. Sự hiểu biết về kiểu dữ liệu (Data types) và các phép biến đổi Tensor là nền tảng để tin tưởng vào kết quả nghiên cứu trong tương lai.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Negation tuning trên nơ-ron QVK dựa trên `aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md`. Sử dụng `torch.split` để phân tích đa kênh và so sánh tính phân tán với MLP.
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
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 1)](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) | [Xem bài viết →](aero_LLM_20_CodeChallenge Negation tuning in MLP neurons (part 1).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 2)](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) | [Xem bài viết →](aero_LLM_21_CodeChallenge Negation tuning in MLP neurons (part 2).md) |
| [Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) | [Xem bài viết →](aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md) |
| 📌 **[Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron QVK (Attention)](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md)** | [Xem bài viết →](aero_LLM_23_CodeChallenge Negation tuning in QVK neurons.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
