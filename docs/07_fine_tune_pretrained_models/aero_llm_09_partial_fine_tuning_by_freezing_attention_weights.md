
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [07 fine tune pretrained models](index.md)

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
# Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM

## Tóm tắt

Bài viết này nghiên cứu phương pháp tinh chỉnh từng phần (partial fine-tuning) các mô hình ngôn ngữ lớn (LLMs) thông qua việc đóng băng (freezing) cơ chế Self-Attention và chỉ cập nhật các lớp Feed-Forward (MLP) và Layer Normalization. Dựa trên dữ liệu thực nghiệm từ thử thách "Partial fine-tuning by freezing attention weights", nghiên cứu phân tích tác động của chiến lược này đến tốc độ huấn luyện, bộ nhớ GPU và khả năng thích nghi phong cách văn học. Kết quả cho thấy việc đóng băng Attention giúp giảm đáng kể số lượng tham số cần cập nhật mà vẫn duy trì được hiệu quả học tập tương đương với tinh chỉnh toàn phần trong các tác vụ hẹp.

---

## 1. Giới thiệu

Fine-tuning toàn bộ (Full Fine-tuning) một mô hình Transformer đòi hỏi tài nguyên tính toán cực lớn. Để tối ưu hóa, các kỹ thuật tinh chỉnh hiệu quả tham số (Parameter-Efficient Fine-Tuning - PEFT) đã ra đời.

Theo tài liệu thực nghiệm , một trong những phương pháp đơn giản nhưng hiệu quả là "Partial Fine-tuning". Thay vì cập nhật toàn bộ 125 triệu tham số (đối với GPT-Neo 125M), chúng ta có thể đóng băng các thành phần đã học tốt các mối liên kết ngôn ngữ toàn cục – cụ thể là cơ chế Attention – và tập trung vào các lớp MLP, nơi chứa đựng phần lớn tri thức về các đặc trưng cụ thể của dữ liệu.

Mục tiêu nghiên cứu:
* Phân tích cơ chế đóng băng trọng số trong kiến trúc Transformer.
* Đo lường tỷ lệ tham số được huấn luyện so với tổng số tham số.
* Đánh giá hiệu quả ổn định gradient và hội tụ của hàm mất mát.

---

## 2. Cơ sở lý thuyết

### 2.1. Cấu trúc Transformer Block

Mỗi block Transformer gồm hai thành phần chính:
1. **Multi-Head Self-Attention (MSA):** Học các quan hệ ngữ cảnh giữa các token.
2. **Multi-Layer Perceptron (MLP):** Thực hiện biến đổi phi tuyến các đặc trưng.

Đầu ra của một block:

$$
h' = \text{LayerNorm}(x + \text{MSA}(x))
$$

$$
y = \text{LayerNorm}(h' + \text{MLP}(h'))
$$

---

### 2.2. Cơ chế đóng băng tham số (Freezing)

Khi đóng băng một lớp, chúng ta đặt thuộc tính:

$$
\text{requires\_grad} = \text{False}
$$

Điều này dẫn đến việc bỏ qua tính toán gradient cho các tham số đó trong quá trình lan truyền ngược (backpropagation):

$$
\frac{\partial \mathcal{L}}{\partial W_{attention}} = 0
$$

---

### 2.3. Tỷ lệ tham số huấn luyện

Nếu gọi $P_{total}$ là tổng tham số và $P_{trainable}$ là tham số được cập nhật:

$$
R = \frac{P_{trainable}}{P_{total}}
$$

Trong bài toán đóng băng Attention, tỷ lệ này thường dao động quanh mức 0.5 (tương đương 50% tham số), giúp tiết kiệm đáng kể tài nguyên GPU.

---

## 3. Phương pháp nghiên cứu

### 3.1. Thiết lập thí nghiệm

* **Mô hình gốc:** EleutherAI/gpt-neo-125M.
* **Chiến lược:** 
    * Đóng băng tất cả các lớp `Attention`.
    * Đóng băng các `Embedding` layers.
    * Chỉ cho phép huấn luyện các lớp `Linear` trong MLP và các lớp `LayerNorm`.
* **Dữ liệu:** Văn bản phong cách Alice và Edgar.

---

### 3.2. Quy trình thực hiện

1. Nạp mô hình tiền huấn luyện.
2. Duyệt qua tất cả các tham số (`named_parameters`).
3. Kiểm tra tên tham số (`"attn"` hoặc `"embed"`).
4. Thiết lập `requires_grad = False` cho các tham số trùng khớp.
5. Khởi tạo Optimizer (chỉ nạp các tham số có `requires_grad = True`).

---

## 4. Kết quả thực nghiệm

### 4.1. Phân tích số lượng tham số

Theo dữ liệu từ , kết quả thống kê cho thấy:
* Tổng tham số: ~125,000,000.
* Tham số huấn luyện sau khi đóng băng Attention: ~65,000,000.
* **Tỷ lệ giảm:** Gần 48%.

---

### 4.2. Khả năng hội tụ

Mặc dù đóng băng một phần quan trọng của mô hình, đồ thị hàm mất mát ($\mathcal{L}$) vẫn cho thấy xu hướng giảm ổn định:

$$
\lim_{t \to \infty} \mathcal{L}(t) = \mathcal{L}_{min}
$$

Đặc biệt, việc đóng băng Attention giúp giảm hiện tượng "catastrophic forgetting" (quên kiến thức cũ), vì các cấu trúc ngôn ngữ cơ bản trong Attention được giữ nguyên.

---

### 3.3. Hiệu năng tính toán

* **Bộ nhớ GPU:** Giảm khoảng 25-30% do không cần lưu trữ trạng thái optimizer (moments) cho các trọng số Attention.
* **Tốc độ:** Tăng nhẹ do giảm số lượng phép tính cập nhật trọng số.

---

## 5. Thảo luận

### 5.1. Tại sao lại đóng băng Attention?

Cơ chế Attention của các mô hình tiền huấn luyện đã rất mạnh trong việc hiểu cấu trúc câu và quan hệ ngữ pháp. Trong khi đó, các lớp MLP thường chịu trách nhiệm "ghi nhớ" các sự kiện hoặc đặc trưng cụ thể của miền dữ liệu (domain-specific knowledge). Vì vậy, tinh chỉnh MLP là đủ để mô hình học phong cách mới.

---

### 5.2. So sánh với LoRA

Trong khi LoRA thêm các ma trận bổ sung, "Partial Fine-tuning" trực tiếp sử dụng các tham số có sẵn. Đây là phương pháp "PEFT sơ khai" nhưng cực kỳ ổn định và không làm tăng độ trễ khi suy luận (inference latency).

---

## 6. Kết luận

Tinh chỉnh từng phần bằng cách đóng băng trọng số Attention là một chiến lược hiệu quả để tối ưu hóa quá trình huấn luyện LLM. Nó cung cấp sự cân bằng giữa hiệu năng (accuracy) và chi phí (computation). Đối với các nhiệm vụ chuyển đổi phong cách văn học như Alice-Edgar, phương pháp này chứng minh rằng chúng ta không cần cập nhật toàn bộ mô hình để đạt được kết quả mong muốn.

---

## Tài liệu tham khảo

1. Tài liệu thực nghiệm: Partial fine-tuning by freezing attention weights.
2. Vaswani et al. (2017). *Attention Is All You Need*.
3. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
4. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 07_fine_tune_pretrained_models](README.md) | [Xem bài viết →](README.md) |
| [Fine-tuning Có Mục Tiêu và Đóng Băng Chính Xác Trọng Số Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_010_codechallenge_fine_tuning_and_targeted_freezing_part_1_.md) | [Xem bài viết →](aero_llm_010_codechallenge_fine_tuning_and_targeted_freezing_part_1_.md) |
| [Phân Tích Hiệu Quả Fine-tuning và Targeted Freezing (Phần 2): Đánh Giá Bằng Trực Quan Hóa và Chuẩn Ma Trận](aero_llm_011_codechallenge_fine_tuning_and_targeted_freezing_part_2_.md) | [Xem bài viết →](aero_llm_011_codechallenge_fine_tuning_and_targeted_freezing_part_2_.md) |
| [Fine-tuning Hiệu Quả Tham Số (Parameter-Efficient Fine-Tuning – PEFT) Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_012_parameter_efficient_fine_tuning_peft_.md) | [Xem bài viết →](aero_llm_012_parameter_efficient_fine_tuning_peft_.md) |
| [Mô Hình CodeGen Cho Bài Toán Hoàn Thành Mã Nguồn: Kiến Trúc, Huấn Luyện và Ứng Dụng](aero_llm_013_codegen_for_code_completion.md) | [Xem bài viết →](aero_llm_013_codegen_for_code_completion.md) |
| [Fine-tuning Mô Hình CodeGen Cho Bài Toán Giải Tích: Phương Pháp, Đánh Giá và Ứng Dụng](aero_llm_014_codechallenge_fine_tune_codegen_for_calculus.md) | [Xem bài viết →](aero_llm_014_codechallenge_fine_tune_codegen_for_calculus.md) |
| [Tinh Chỉnh Mô Hình BERT Cho Bài Toán Phân Loại Cảm Xúc Văn Bản IMDb](aero_llm_015_fine_tuning_bert_for_classification.md) | [Xem bài viết →](aero_llm_015_fine_tuning_bert_for_classification.md) |
| [📘 Ứng Dụng Mô Hình BERT Trong Phân Tích Cảm Xúc Đánh Giá Phim IMDB](aero_llm_016_codechallenge_imdb_sentiment_analysis_using_bert_en_us.md) | [Xem bài viết →](aero_llm_016_codechallenge_imdb_sentiment_analysis_using_bert_en_us.md) |
| [📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) | [Xem bài viết →](aero_llm_017_gradient_clipping_and_learning_rate_scheduler_part_1_en_us.md) |
| [📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) | [Xem bài viết →](aero_llm_018_gradient_clipping_and_learning_rate_scheduler_part_2_.md) |
| [📘 Kết Hợp Gradient Clipping, Freezing và Learning Rate Scheduler Trong Fine-Tuning Mô Hình BERT](aero_llm_019_codechallenge_clip_freeze_and_schedule_bert.md) | [Xem bài viết →](aero_llm_019_codechallenge_clip_freeze_and_schedule_bert.md) |
| [Tối Ưu Hóa Quá Trình Tiền Huấn Luyện Mô Hình Ngôn Ngữ Lớn: Phân Tích Các Chiến Lược Tính Toán và Học Tập](aero_llm_01_what_does_fine_tuning_mean.md) | [Xem bài viết →](aero_llm_01_what_does_fine_tuning_mean.md) |
| [Lưu Trữ và Tải Lại Mô Hình Học Sâu Trong PyTorch và Hugging Face: Phương Pháp, Cấu Trúc và Đánh Giá](aero_llm_020_saving_and_loading_trained_models.md) | [Xem bài viết →](aero_llm_020_saving_and_loading_trained_models.md) |
| [Ứng Dụng Mô Hình BERT Trong Phân Loại Văn Bản Văn Học: Trường Hợp Alice và Edgar](aero_llm_021_bert_decides_alice_or_edgar.md) | [Xem bài viết →](aero_llm_021_bert_decides_alice_or_edgar.md) |
| [Đồng Tiến Hóa Mô Hình Sinh Văn Bản và Mô Hình Phân Loại: Trường Hợp Alice và Edgar](aero_llm_022_codechallenge_evolution_of_alice_and_edgar_part_1_.md) | [Xem bài viết →](aero_llm_022_codechallenge_evolution_of_alice_and_edgar_part_1_.md) |
| [📘 Đánh Giá Mô Hình Sinh Văn Bản Thông Qua Phân Loại BERT: Nghiên Cứu Trường Hợp Alice và Edgar](aero_llm_023_codechallenge_evolution_of_alice_and_edgar_part_2_.md) | [Xem bài viết →](aero_llm_023_codechallenge_evolution_of_alice_and_edgar_part_2_.md) |
| [Fine-tuning Mô hình GPT-2 trên Tác phẩm *Gulliver’s Travels*: Phân tích Thực nghiệm và Đánh giá Hiệu quả](aero_llm_02_fine_tune_a_pretrained_gpt2.md) | [Xem bài viết →](aero_llm_02_fine_tune_a_pretrained_gpt2.md) |
| [Đánh giá Ảnh hưởng của Learning Rate trong Fine-tuning GPT-2 trên *Gulliver’s Travels*](aero_llm_03codechallenge_gulliver_s_learning_rates.md) | [Xem bài viết →](aero_llm_03codechallenge_gulliver_s_learning_rates.md) |
| [Nghiên cứu Quy trình Sinh Văn bản từ Mô hình Ngôn ngữ Tiền Huấn luyện GPT-2](aero_llm_04_on_generating_text_from_pretrained_models.md) | [Xem bài viết →](aero_llm_04_on_generating_text_from_pretrained_models.md) |
| [Tinh Chỉnh Mô Hình GPT-2 Bằng Hàm Mất Mát KL Divergence Để Tối Ưu Hóa Việc Sinh Token Chứa Ký Tự “X”](aero_llm_05_codechallenge_maximize_the_x_factor_.md) | [Xem bài viết →](aero_llm_05_codechallenge_maximize_the_x_factor_.md) |
| [Tinh Chỉnh Mô Hình GPT-Neo Để Mô Phỏng Phong Cách Văn Học Alice in Wonderland và Edgar Allan Poe](aero_llm_06_alice_in_wonderland_and_edgar_allen_poe_with_gpt_neo_.md) | [Xem bài viết →](aero_llm_06_alice_in_wonderland_and_edgar_allen_poe_with_gpt_neo_.md) |
| [Đánh Giá Định Lượng và Định Tính Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp Văn Phong *Alice* và *Edgar Allan Poe*](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) |
| [Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) |
| [Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) | [Xem bài viết →](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) |
| 📌 **[Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md)** | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
