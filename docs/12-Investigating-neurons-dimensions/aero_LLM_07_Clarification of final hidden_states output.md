
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
# Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)

## Tóm tắt (Abstract)
Báo cáo này giải quyết một sự khác biệt quan trọng trong việc trích xuất hoạt hóa giữa phương pháp Hooks và `output.hidden_states` tại tầng cuối cùng của mô hình Transformer (GPT-2). Trong khi ở các tầng trung gian, hai phương pháp này cho kết quả trùng khớp hoàn hảo, thì tại tầng cuối cùng, giá trị trích xuất từ Hidden States đã được đi qua một lớp chuẩn hóa bổ sung gọi là Final LayerNorm ($L_f$). Nghiên cứu thực nghiệm chứng minh sự khác biệt này và giải thích lý do tại sao các mẫu tính toán ở tầng cuối cùng thường mang các đặc tính định lượng khác biệt so với phần còn lại của mô hình.

---

## 1. Mở Đầu (Introduction)
Trong các bài báo trước, chúng ta đã giả định rằng `hidden_states[i]` tương đương với đầu ra của Transformer Block thứ `i-1`. Tuy nhiên, khi đi sâu vào phân tích cơ học, chúng ta phát hiện một ngoại lệ tại điểm kết thúc của residual stream. Việc hiểu rõ ranh giới giữa Khối Transformer cuối cùng và lớp chuẩn hóa cuối cùng là tối quan trọng để giải mã chính xác các biểu diễn trước khi chúng được chuyển đổi thành Logits.

---

## 2. Phẫu thuật Kiến trúc: Transformer Block và Final LayerNorm

### 2.1. Cấu trúc chuẩn của Hidden States
Đối với các tầng từ $0$ đến $N-2$ (với $N$ là tổng số tầng):
- **Hook Output:** Giá trị hoạt hóa ngay sau lớp MLP Projection.
- **Hidden State Output:** Trùng khớp $100\%$ với Hook Output.

### 2.2. Sự khác biệt tại tầng $N-1$
Tại khối Transformer cuối cùng:
- **Hook Output:** Là kết quả của MLP cuối cùng cộng vào residual stream.
- **Hidden State Output:** Là Hook Output đã được đẩy qua `model.transformer.ln_f`.

---

## 3. Thực Nghiệm Đối Chứng (Experimental Verification)

### 3.1. Thử nghiệm "Penultimate vs. Final"
Bằng cách sử dụng phép trừ ma trận giữa dữ liệu Hook và dữ liệu Hidden States:
1. **Tại tầng áp chót (Penultimate):** Hiệu số bằng 0 tuyệt đối. Điều này xác nhận sự đồng nhất của hai phương pháp trích xuất ở các tầng giữa.
2. **Tại tầng cuối (Final):** Hiệu số khác 0 đáng kể. Điều này chỉ ra rằng có một phép biến đổi toán học đã xảy ra giữa điểm trích xuất của Hook và điểm trích xuất của Hidden States.

### 3.2. Chứng minh bằng Final LayerNorm
Khi lấy kết quả từ Hook tại tầng cuối và chủ động đẩy nó qua lớp `model.transformer.ln_f`, hiệu số so với `hidden_states[-1]` trở về bằng 0. Đây là bằng chứng thực nghiệm khẳng định rằng Hidden State cuối cùng thực chất là một trạng thái "đã chuẩn hóa" để chuẩn bị cho bước nhân với ma trận nhúng đầu ra (un-embedding).

---

## 4. Thảo Luận: Tại sao tầng cuối cùng lại "đặc biệt"?
Nhà nghiên cứu cần lưu ý hai lý do khiến dữ liệu tầng cuối thường trông khác biệt trên đồ thị:
1. **Áp lực tính toán:** Đây là cơ hội cuối cùng để mô hình tinh chỉnh vector dự báo, do đó các nơ-ron có xu hướng hoạt động với cường độ và tính chọn lọc cao hơn.
2. **Biến đổi toán học:** Sự hiện diện của Final LayerNorm làm nén các giá trị hoạt hóa về một vùng phân phối ổn định hơn, che lấp đi các biến động biên độ cực lớn thường thấy trong dòng dư chưa chuẩn hóa.

---

## 5. Kết Luận
Báo cáo khẳng định rằng khi thực hiện các phân tích so sánh xuyên tầng (laminar profile), cần phải đồng nhất phương pháp trích xuất. Nếu sử dụng `output.hidden_states`, hãy nhớ rằng tầng cuối cùng đã được chuẩn hóa. Nếu muốn quan sát "tư duy thô" (raw thinking) của mô hình ở tầng cuối, sử dụng Hooks là lựa chọn tối ưu hơn.

---

## Tài liệu tham khảo (Citations)
1. Phân tích sự khác biệt cơ học tại tầng cuối cùng của GPT-2 dựa trên `aero_LLM_07_Clarification of final hidden_states output.md`. Xác minh vai trò của `ln_f` đối với Hidden States.
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
| 📌 **[Làm rõ về Hidden States Tầng cuối: Vai trò của LayerNorm (Clarification of Final Hidden States)](aero_LLM_07_Clarification of final hidden_states output.md)** | [Xem bài viết →](aero_LLM_07_Clarification of final hidden_states output.md) |
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
