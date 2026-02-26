
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
# Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar

## Tóm tắt

Bài viết này tập trung vào phương pháp định lượng sự thay đổi phong cách văn bản sau quá trình fine-tuning mô hình ngôn ngữ GPT-Neo. Sử dụng bộ phân loại BERT như một công cụ đo lường khách quan, nghiên cứu này đánh giá mức độ hội tụ của mô hình sinh văn bản về phía hai tác giả mục tiêu: Lewis Carroll (Alice) và Edgar Allan Poe (Edgar). Kết quả cho thấy các chỉ số định lượng như độ chính xác phân loại (Classification Accuracy) và hàm mất mát (Loss) là những công cụ phản ánh chính xác tiến trình học tập của mô hình.

---

## 1. Giới thiệu

Việc đánh giá tính sáng tạo và phong cách của các mô hình ngôn ngữ sau tinh chỉnh thường mang tính định tính và cảm tính. Để đưa ra những đánh giá khoa học hơn, chúng ta cần các phương pháp định lượng.

Theo tài liệu , thử thách "Quantify the Alice-Edgar fine-tuning" được thiết kế để đo lường xem mô hình sinh văn bản đã học được bao nhiêu tri thức về phong cách văn học mục tiêu thông qua một bộ phân loại độc lập.

Mục tiêu nghiên cứu:
* Xây dựng hệ thống đo lường hiệu quả tinh chỉnh.
* Phân tích sự thay đổi của độ chính xác phân loại theo thời gian.
* Đánh giá mối tương quan giữa sự hội tụ của mô hình sinh và mô hình phân loại.

---

## 2. Cơ sở lý thuyết

### 2.1. Đo lường sự khác biệt phân phối

Quá trình tinh chỉnh nhằm mục đích đưa phân phối xác suất của mô hình sinh ($P_{model}$) tiến gần đến phân phối xác suất của dữ liệu mục tiêu ($P_{data}$):
$$
D_{KL}(P_{data} \parallel P_{model}) \rightarrow 0
$$


Trong bài toán này, chúng ta sử dụng một bộ phân loại $C$ để ước lượng xác suất hậu nghiệm:
$$
\hat{y} = C(x) = P(\text{Style} \mid x)
$$


---

### 2.2. Chỉ số định lượng

Hai chỉ số chính được sử dụng để đánh giá:

1. **Độ chính xác phân loại (Accuracy):**
$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\arg\max C(x_i) = y_i)
$$


2. **Hàm mất mát Cross-Entropy (Log-Loss):**
$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$


---

## 3. Quy trình thực nghiệm

### 3.1. Thiết lập mô hình

* **Mô hình sinh:** GPT-Neo 125M được tinh chỉnh trên hai tập dữ liệu khác nhau.
* **Bộ phân loại:** BERT (base) đã được huấn luyện trước trên dữ liệu văn học Alice và Edgar.
* **Tập dữ liệu kiểm tra:** 121 đoạn văn bản chưa được sử dụng trong quá trình huấn luyện.

---

### 3.2. Chu kỳ đánh giá

Theo , việc đánh giá không thực hiện liên tục để tiết kiệm tài nguyên. Thay vào đó, sau mỗi 10 batch huấn luyện, mô hình sinh sẽ tạo ra các đoạn văn bản mẫu và bộ phân loại BERT sẽ tiến hành gán nhãn.

Tiến trình:
$$
t = \{10, 20, 30, \dots, T\}
$$


---

## 4. Phân tích kết quả

### 4.1. Sự tăng trưởng của độ chính xác

Tại giai đoạn đầu huấn luyện ($t=0$), bộ phân loại BERT gặp khó khăn trong việc phân biệt văn bản sinh từ hai mô hình, độ chính xác dao động quanh mức ngẫu nhiên:
$$
\text{Acc}_{t=0} \approx 0.5
$$


Khi quá trình tinh chỉnh tiến triển, văn bản sinh bắt đầu mang các đặc trưng phong cách rõ rệt hơn, dẫn đến độ chính xác tăng nhanh:
$$
\text{Acc}_{t \rightarrow T} \rightarrow 0.9
$$


---

### 4.2. Biểu đồ hội tụ

Quan hệ giữa Loss của mô hình phân loại trên văn bản sinh và số bước huấn luyện:
$$
\frac{\partial \mathcal{L}_{cls}}{\partial t} < 0
$$


Điều này xác nhận rằng mô hình sinh đang thực sự "di chuyển" trong không gian đặc trưng về phía vùng dữ liệu của Alice hoặc Edgar.

---

## 5. Thảo luận

### 5.1. Ưu điểm của phương pháp định lượng

* **Khách quan:** Loại bỏ yếu tố thiên kiến của con người trong đánh giá văn bản.
* **Thời gian thực:** Cho phép giám sát quá trình huấn luyện và dừng sớm (early stopping) khi đạt yêu cầu.
* **Tính quy mô:** Có thể áp dụng để đánh giá hàng nghìn mẫu văn bản trong thời gian ngắn.

---

### 5.2. Các yếu tố gây nhiễu

* **Sự sai lệch của Tokenizer:** Việc ánh xạ token giữa GPT-Neo và BERT có thể gây mất mát thông tin.
* **Chất lượng bộ phân loại:** Nếu BERT chưa được huấn luyện tốt, kết quả định lượng sẽ không còn tin cậy.

---

## 6. Kết luận

Thử thách định lượng quá trình tinh chỉnh Alice và Edgar đã chứng minh tính hiệu quả của việc sử dụng mô hình AI để đánh giá mô hình AI. Việc kết hợp giữa các chỉ số toán học và mô hình phân loại sâu cung cấp một cái nhìn toàn diện và chính xác về khả năng học phong cách của các LLM hiện đại.

---

## Tài liệu tham khảo

1. Tài liệu hướng dẫn: CodeChallenge Quantify the AliceEdgar fine-tuning.
2. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.
3. Chen et al. (2021). *Evaluating Large Language Models for Code*.
4. Goodfellow et al. (2016). *Deep Learning*.

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
| 📌 **[Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md)** | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) |
| [Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) | [Xem bài viết →](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) |
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
