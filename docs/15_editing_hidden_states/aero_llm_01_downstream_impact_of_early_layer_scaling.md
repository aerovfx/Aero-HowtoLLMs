
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [15 editing hidden states](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Tác động Hạ nguồn của việc Thay đổi Quy mô Lớp sớm (Downstream Impact of Early Layer Scaling)

## Tóm tắt (Abstract)
Nghiên cứu này khảo sát sự lan truyền nhiễu và khả năng phục hồi của mô hình GPT-2 khi đối mặt với các can thiệp nhân quả tại các lớp sớm. Bằng cách sử dụng cơ chế Forward Hook để thay đổi quy mô (Scaling) của Hidden States tại Transformer Block thứ 3 (Layer 2), báo cáo phân tích sự sai biệt giữa trạng thái "nguyên bản" và trạng thái "bị can thiệp" thông qua chỉ số chuẩn ma trận (Matrix Norm). Kết quả thực nghiệm cho thấy sự sai lệch tín hiệu không những không bị triệt tiêu mà còn có xu hướng tích tụ và khuếch đại khi đi sâu hơn vào mô hình, ngoại trừ một nỗ lực nén nhẹ tại lớp cuối cùng. Điều này khẳng định tính phụ thuộc nhân quả chặt chẽ của các lớp hạ nguồn đối với dữ liệu từ các lớp thượng nguồn.

---

## 1. Mở Đầu (Introduction)
Trong diễn giải cơ học nhân quả, một nguyên lý cơ bản là: can thiệp vào tầng $n$ sẽ ảnh hưởng đến mọi tầng $n+m$ ($m>0$) nhưng tuyệt đối không tác động ngược lại các tầng trước đó. Thí nghiệm này được thiết kế để đo lường định lượng "độ nhạy" của mô hình đối với các biến động tại những nút thắt cổ chai đầu tiên. Chúng ta đặt ra câu hỏi: Liệu Transformer có cơ chế tự thích nghi để đưa các giá trị bị co giãn về mức bình thường sau vài bước tính toán, hay sự sai lệch sẽ dẫn đến một chuỗi sụp đổ phản ứng theo dây chuyền?

---

## 2. Tiết Thiết Lập Can Thiệp (Methodology)

### 2.1. Cơ chế Hook và Biến Toàn Cục
Sử dụng một hàm Hook đơn giản để can thiệp vào đầu ra của Transformer Block. Điểm mấu chốt là việc sử dụng biến số `scaling_factor` được định nghĩa ở phạm vi toàn cục (Global scope).
- **Quy trình:** Trích xuất Tuple Output $\to$ Lấy phần tử đầu tiên (Hidden State Tensor) $\to$ Sử dụng phương thức nhân tại chỗ (In-place multiplication) `.mul_()` để tiết kiệm bộ nhớ $\to$ Đóng gói lại vào Tuple trước khi trả về.
- **Biến đổi:** Việc thay đổi `scaling_factor` từ bên ngoài hàm Hook cho phép thực hiện nhiều kịch bản (ví dụ: giảm 50% hoặc tăng 150%) mà không cần định nghĩa lại cấu trúc Hook, giúp tối ưu hóa luồng thí nghiệm.

### 2.2. Chỉ số Đo lường Sai biệt (Matrix Norm Difference)
Để đo lường tác động, ta tính toán hiệu số giữa Hidden States sạch ($\mathbf{H}_{pure}$) và Hidden States bị can thiệp ($\mathbf{H}_{scaled}$):

$$
\Delta = \|\mathbf{H}_{pure} - \mathbf{H}_{scaled}\|_F
$$

Trong đó $\\mid\cdot\\mid_F$ là chuẩn Frobenius của ma trận. Nếu $\Delta = 0$, can thiệp không gây ra sự thay đổi. Giá trị $\Delta$ càng lớn chứng tỏ mô hình càng đi chệch khỏi quỹ đạo tính toán ban đầu.

---

## 3. Khảo Sát Tác Động Hạ Nguồn (Analysis)

### 3.1. Sự Hiện Diện Của Sai Lệch (The Divergence)
Đồ thị biểu diễn sai biệt cho thấy tại các tầng trước can thiệp (Embedding, Layer 0, Layer 1), $\Delta \approx 0$. Ngay tại Layer 2, giá trị này nhảy vọt. Đáng chú ý, từ Layer 3 trở đi, $\Delta$ liên tục tăng trưởng một cách phi tuyến. Điều này bác bỏ giả thuyết về việc mô hình có thể "tự chữa lành" hoàn toàn sự co giãn tín hiệu chỉ bằng các lớp Normalization hạ nguồn.

### 3.2. Hiện Tượng Token Đầu Tiên (The First Token Quirk)
Một phát hiện thực nghiệm quan trọng: Nếu bao gồm cả Token đầu tiên trong phân tích, giá trị sai biệt $\Delta$ sẽ bùng nổ vượt tầm kiểm soát (tăng vọt một bậc độ lớn - Order of magnitude). Hiện tượng này xảy ra do Token đầu tiên thường mang các đặc tính khởi tạo hoặc dấu hiệu phân đoạn (BOS) có biên độ cực lớn. Khuyến nghị nghiên cứu: Luôn loại bỏ Token đầu tiên khỏi các tác vụ đo lường nhân quả để tránh nhiễu hệ thống.

### 3.3. Phản Ứng Tại Tầng Cuối Cùng
Quan sát phổ biến trên nhiều hệ số Scale (từ 0.5 đến 1.5): Tại Transformer Block cuối cùng (ngay trước khi vào Embedding Matrix đầu ra), có một sự sụt giảm nhẹ của $\Delta$. Điều này gợi ý rằng tầng cuối cùng thực hiện một nhiệm vụ "hiệu chỉnh" (Calibrating), cố gắng nén các giá trị hoạt hóa về một vùng phân phối ổn định hơn để chuẩn bị cho bước sinh từ.

---

## 4. Kết Luận
Thí nghiệm minh chứng rằng mô hình ngôn ngữ lớn cực kỳ nhạy cảm với các biến đổi sơ khởi. Một can thiệp đơn giản như giảm quy mô tín hiệu xuống 50% tại một lớp sớm sẽ gây ra sự sai lệch ngày càng lớn dọc theo residual stream. Khả năng "bù trừ" của mô hình là có tồn tại nhưng rất hạn chế và chỉ tập trung ở giai đoạn cuối cùng. Điều này đặt ra nền tảng cho việc nghiên cứu sâu hơn về cách các tầng cụ thể "định hình" nội dung ngôn ngữ và khả năng dự báo của mô hình.

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm về Downstream impact của Hidden State scaling được trình bày trong `aero_LLM_01_Downstream impact of early layer scaling.md`. Giải phẫu hiện tượng "nổ" sai biệt tại Token đầu tiên.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Tác động Hạ nguồn của việc Thay đổi Quy mô Lớp sớm (Downstream Impact of Early Layer Scaling)](aero_llm_01_downstream_impact_of_early_layer_scaling.md)** | [Xem bài viết →](aero_llm_01_downstream_impact_of_early_layer_scaling.md) |
| [Thử thách Lập trình: Thay đổi Quy mô Hidden State và Tổn thất Token](aero_llm_02_codechallenge_hidden_state_scaling_and_token_loss.md) | [Xem bài viết →](aero_llm_02_codechallenge_hidden_state_scaling_and_token_loss.md) |
| [Thử thách Lập trình: Dự đoán BERT với Nhiễu và Hoán vị (Noisy and Shuffled BERT Predictions)](aero_llm_03_codechallenge_noisy_and_shuffled_bert_predictions.md) | [Xem bài viết →](aero_llm_03_codechallenge_noisy_and_shuffled_bert_predictions.md) |
| [Thử thách Lập trình: Đo lường và Hiệu chỉnh Định kiến Giới trong BERT](aero_llm_04_codechallenge_measure_and_correct_bert_s_bias.md) | [Xem bài viết →](aero_llm_04_codechallenge_measure_and_correct_bert_s_bias.md) |
| [Vá Hoạt hóa và Tác vụ Nhận diện Tân ngữ Gián tiếp (Activation Patching and Indirect Object Identification)](aero_llm_05_activation_patching_with_indirect_object_identification.md) | [Xem bài viết →](aero_llm_05_activation_patching_with_indirect_object_identification.md) |
| [Bỏ qua một Tầng Transformer (Skip a Layer)](aero_llm_06_skip_a_layer.md) | [Xem bài viết →](aero_llm_06_skip_a_layer.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
