
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [06 pretraining](index.md)

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
Dưới đây là bài viết khoa học được biên soạn dựa trên tài liệu đính kèm, có bổ sung trích dẫn và trình bày dưới dạng **Markdown**.

---

# Phân Tích Hành Vi Học Biểu Diễn Token Trong Mô Hình Ngôn Ngữ Lớn

## Tóm tắt (Abstract)

Bài viết này nghiên cứu cách mô hình ngôn ngữ lớn (Large Language Models – LLMs) học và xử lý các token có tần suất xuất hiện khác nhau trong dữ liệu huấn luyện. Dựa trên chuỗi thí nghiệm sử dụng văn bản *Gulliver’s Travels* và kiến trúc tương tự GPT-2, nghiên cứu cho thấy rằng mô hình không chỉ học từ các token xuất hiện thường xuyên mà còn điều chỉnh xác suất của các token hiếm và không xuất hiện. Kết quả cho thấy cơ chế softmax và lan truyền ngược giúp mô hình cập nhật toàn bộ không gian embedding, ngay cả với các token không xuất hiện trực tiếp trong chuỗi đầu vào.

---

## 1. Giới thiệu (Introduction)

Mô hình ngôn ngữ lớn dựa trên Transformer đã trở thành nền tảng cho nhiều ứng dụng xử lý ngôn ngữ tự nhiên hiện đại. Tuy nhiên, cách mà các mô hình này học và biểu diễn các token hiếm hoặc không xuất hiện vẫn chưa được hiểu rõ.

Một câu hỏi quan trọng đặt ra là:

> Liệu các token hiếm hoặc chưa từng xuất hiện trong dữ liệu huấn luyện có được mô hình “học” hay không?

Nghiên cứu này nhằm trả lời câu hỏi trên thông qua việc phân tích phân phối log-softmax của các nhóm token có tần suất khác nhau.

---

## 2. Cơ sở lý thuyết (Background)

### 2.1 Tokenization và Từ vựng

Trong các mô hình như GPT-2, văn bản được chia thành các token dựa trên Byte Pair Encoding (BPE). Mỗi token tương ứng với một chỉ số trong từ vựng kích thước khoảng 50.000.

### 2.2 Softmax và Log-Softmax

Xác suất dự đoán token được tính bằng:

$$

P(y=i|x) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}

$$


Trong đó $z_i$ là logit của token $i$.

Log-softmax được sử dụng để ổn định số học:

$$

\log P(y=i|x) = z_i - \log \sum_j e^{z_j}

$$


### 2.3 Lan truyền ngược trong LLM

Trong quá trình huấn luyện, gradient được lan truyền qua toàn bộ ma trận embedding, dẫn đến việc cập nhật tham số cho cả những token không xuất hiện trực tiếp trong batch.

---

## 3. Phương pháp nghiên cứu (Methodology)

### 3.1 Dữ liệu

* Văn bản: *Gulliver’s Travels* (Project Gutenberg)
* Tokenizer: GPT-2 tokenizer
* Khoảng 20% từ vựng xuất hiện ít nhất một lần trong văn bản


### 3.2 Phân loại token

Token được chia thành ba nhóm:

1. **Most common tokens**: 100 token xuất hiện nhiều nhất
2. **Least common tokens**: 100 token xuất hiện ít nhất
3. **Never-used tokens**: 100 token không xuất hiện

### 3.3 Mô hình

* Kiến trúc: Transformer tương tự GPT-2
* Không sử dụng dropout
* Đầu ra: log-softmax
* Huấn luyện trên GPU

### 3.4 Quy trình thí nghiệm

#### Bước 1: Phân tích dữ liệu

* Đếm tần suất token
* Phân loại token theo nhóm

#### Bước 2: Mô hình chưa huấn luyện

* Truyền dữ liệu qua mô hình ngẫu nhiên
* Trích xuất log-softmax

#### Bước 3: Loại bỏ nhiễu

* Loại bỏ token xuất hiện trong batch đầu vào
* Áp dụng mask trên output

#### Bước 4: Huấn luyện

* Huấn luyện 500 epoch
* Theo dõi loss và log-softmax trung bình

---

## 4. Kết quả (Results)

### 4.1 Mô hình chưa huấn luyện

Khi mô hình chưa được huấn luyện:

* Log-softmax của ba nhóm xấp xỉ nhau
* Giá trị trung bình khoảng: −10.8 đến −11
* Không có sự phân biệt giữa các token

Điều này phản ánh phân phối ngẫu nhiên ban đầu.

### 4.2 Ảnh hưởng của batch đầu vào

Nếu không loại bỏ token xuất hiện trong batch:

* Token phổ biến có lợi thế
* Kết quả bị nhiễu

Sau khi áp dụng mask:

* So sánh trở nên công bằng
* Phân phối ổn định hơn

### 4.3 Sau huấn luyện

Sau 500 epoch:

| Nhóm Token      | Xu hướng log-softmax | Xác suất |
| --------------- | -------------------- | -------- |
| Phổ biến        | Gần mức ngẫu nhiên   | Cao      |
| Ít gặp          | Giảm dần             | Thấp     |
| Không xuất hiện | Giảm mạnh            | Rất thấp |

Kết quả cho thấy:

* Token phổ biến được “tăng cường”
* Token hiếm bị suy giảm
* Token không xuất hiện bị triệt tiêu mạnh


---

## 5. Thảo luận (Discussion)

### 5.1 Cơ chế lan truyền thông tin

Mặc dù token không xuất hiện trong chuỗi đầu vào, nhưng:

* Softmax phụ thuộc toàn bộ từ vựng
* Gradient ảnh hưởng mọi embedding

Do đó, mô hình vẫn “học gián tiếp” về token hiếm.

### 5.2 Hiệu ứng bất cân bằng dữ liệu

Kết quả phản ánh hiện tượng:

* Token phổ biến chiếm ưu thế
* Chủ đề hiếm bị suy giảm xác suất

Điều này giải thích vì sao LLM:

* Viết tốt nội dung phổ biến (social media, blog)
* Kém chính xác với chủ đề hiếm (lịch sử cổ đại, ngôn ngữ ít tài nguyên)

### 5.3 So sánh không gian log và tuyến tính

Trong không gian tuyến tính:

* Token hiếm → xác suất tiệm cận 0
* Token phổ biến → chiếm phần lớn phân phối

So sánh log-scale giúp quan sát rõ động lực học huấn luyện.

---

## 6. Hệ quả và ứng dụng (Implications)

### 6.1 Đối với huấn luyện LLM

* Cần cân bằng dữ liệu
* Bổ sung dữ liệu hiếm
* Áp dụng kỹ thuật re-weighting

### 6.2 Đối với Fine-tuning

Fine-tuning có thể:

* Phục hồi token hiếm
* Điều chỉnh phân phối xác suất

Tuy nhiên, cần nhiều dữ liệu chuyên biệt.

### 6.3 Đối với thiết kế mô hình

* Cần xem xét kiến trúc adaptive softmax
* Hoặc vocabulary pruning

---

## 7. Kết luận (Conclusion)

Nghiên cứu cho thấy rằng:

1. Mô hình ngôn ngữ học từ toàn bộ từ vựng, không chỉ token xuất hiện.
2. Token phổ biến được ưu tiên trong quá trình huấn luyện.
3. Token hiếm và không xuất hiện bị suy giảm xác suất đáng kể.
4. Cơ chế softmax và backpropagation đóng vai trò trung tâm trong việc lan truyền thông tin.

Kết quả này giúp giải thích hành vi của LLM trong thực tế và cung cấp cơ sở cho việc cải tiến phương pháp huấn luyện trong tương lai.

---

## Tài liệu tham khảo (References)

1. Tài liệu hướng dẫn CodeChallenge: *What happens to unused tokens*


2. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.

3. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI.

4. Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📘 Huấn Luyện Mô Hình Ngôn Ngữ Với Thiên Lệch Có Chủ Đích Bằng KL-Divergence: Một Nghiên Cứu Thực Nghiệm](aero_llm_010_codechallenge_train_a_model_to_like_x.md) | [Xem bài viết →](aero_llm_010_codechallenge_train_a_model_to_like_x.md) |
| [📘 Các Vấn Đề Tỷ Lệ Số Học Trong Mô Hình Học Sâu: Phân Tích Vai Trò Của Scaling và Normalization Trong Cơ Chế Attention](aero_llm_011_codechallenge_numerical_scaling_issues_in_dl_models_copy_2.md) | [Xem bài viết →](aero_llm_011_codechallenge_numerical_scaling_issues_in_dl_models_copy_2.md) |
| [Weight Initialization and Numerical Stability in Large Language Models](aero_llm_012_weight_initializations.md) | [Xem bài viết →](aero_llm_012_weight_initializations.md) |
| [Phân Tích Ảnh Hưởng Của Khởi Tạo Trọng Số Và Sự Tiến Hóa Phân Phối Tham Số Trong Quá Trình Huấn Luyện Mô Hình Transformer](aero_llm_013_codechallenge_train_model_5_with_weight_inits.md) | [Xem bài viết →](aero_llm_013_codechallenge_train_model_5_with_weight_inits.md) |
| [Dropout as a Regularization Mechanism in Large Language Models: Theory, Implementation, and Practical Implications](aero_llm_014_dropout_in_theory_and_in_pytorch.md) | [Xem bài viết →](aero_llm_014_dropout_in_theory_and_in_pytorch.md) |
| [So Sánh Đầu Ra Logits và Log-Softmax Trong Mô Hình Ngôn Ngữ: Tác Động Đến Huấn Luyện và Sinh Văn Bản](aero_llm_015_should_you_output_logits_or_log_softmax_logits_.md) | [Xem bài viết →](aero_llm_015_should_you_output_logits_or_log_softmax_logits_.md) |
| [aero llm 016 the fineweb dataset](aero_llm_016_the_fineweb_dataset.md) | [Xem bài viết →](aero_llm_016_the_fineweb_dataset.md) |
| [Tích Hợp Dropout Trong Mô Hình Ngôn Ngữ Transformer: Phân Tích Trường Hợp Model 5](aero_llm_017_codechallenge_fine_dropout_in_model_5_part_1.md) | [Xem bài viết →](aero_llm_017_codechallenge_fine_dropout_in_model_5_part_1.md) |
| [Chiến Lược Huấn Luyện Dựa Trên Final-Token Loss Trong Mô Hình Transformer: Phân Tích Trường Hợp Model 5 Với Dropout](aero_llm_018_codechallenge_fine_dropout_in_model_5_part_2_.md) | [Xem bài viết →](aero_llm_018_codechallenge_fine_dropout_in_model_5_part_2_.md) |
| 📌 **[Phân Tích Hành Vi Học Biểu Diễn Token Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_019_codechallenge_what_happens_to_unused_tokens_.md)** | [Xem bài viết →](aero_llm_019_codechallenge_what_happens_to_unused_tokens_.md) |
| [📘 Vai Trò Của Pre-training Trong Mô Hình Ngôn Ngữ Lớn: Phân Tích Chi Phí, Hiệu Quả và Tính Ứng Dụng](aero_llm_01_what_is_pretraining.md) | [Xem bài viết →](aero_llm_01_what_is_pretraining.md) |
| [Tối Ưu Hóa Quá Trình Tiền Huấn Luyện Mô Hình Ngôn Ngữ Lớn: Phân Tích Các Chiến Lược Tính Toán và Học Tập](aero_llm_020_optimization_options.md) | [Xem bài viết →](aero_llm_020_optimization_options.md) |
| [📘 Nền Tảng Hugging Face Trong Hệ Sinh Thái Trí Tuệ Nhân Tạo: Vai Trò, Cấu Trúc và Ứng Dụng Trong Nghiên Cứu Mô Hình Ngôn Ngữ](aero_llm_02_huggingface.md) | [Xem bài viết →](aero_llm_02_huggingface.md) |
| [📘 Thuật Toán Tối Ưu AdamW Trong Huấn Luyện Mô Hình Học Sâu: Cơ Sở Lý Thuyết, Cải Tiến và Ứng Dụng](aero_llm_03_the_adamw_optimizer.md) | [Xem bài viết →](aero_llm_03_the_adamw_optimizer.md) |
| [📘 So Sánh SGD, Adam và AdamW Trong Huấn Luyện Mô Hình Học Sâu: Phân Tích Thực Nghiệm và Ứng Dụng](aero_llm_04_codechallenge_sgd_vs_adam_vs_adamw_.md) | [Xem bài viết →](aero_llm_04_codechallenge_sgd_vs_adam_vs_adamw_.md) |
| [📘 Huấn Luyện Mô Hình Ngôn Ngữ Đơn Giản Bằng PyTorch: Phân Tích Quy Trình, Động Lực Học và Hiệu Suất Thực Nghiệm](aero_llm_05_train_model.md) | [Xem bài viết →](aero_llm_05_train_model.md) |
| [📘 Thiết Lập Tập Kiểm Thử Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Phương Pháp Train–Test Split và Đánh Giá Hiệu Suất](aero_llm_06_codechallenge_add_a_test_set.md) | [Xem bài viết →](aero_llm_06_codechallenge_add_a_test_set.md) |
| [📘 Chuyển Giao Trọng Số và Đóng Băng Tham Số Trong Huấn Luyện Mô Hình Ngôn Ngữ: Phân Tích Thực Nghiệm Với Embedding GPT-2](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) | [Xem bài viết →](aero_llm_07_codechallenge_train_model_1_with_gpt2_s_embeddings.md) |
| [📘 Phương Pháp Lấy Mẫu Ngẫu Nhiên và Huấn Luyện Mô Hình GPT-2 Thu Gọn: Phân Tích Thực Nghiệm Với Dữ Liệu Văn Bản Cổ Điển](aero_llm_08_codechallenge_train_model_5_with_modifications.md) | [Xem bài viết →](aero_llm_08_codechallenge_train_model_5_with_modifications.md) |
| [Thiết Kế Hàm Mất Mát Tùy Biến Trong Huấn Luyện Mô Hình Ngôn Ngữ Lớn](aero_llm_09_create_a_custom_loss_function.md) | [Xem bài viết →](aero_llm_09_create_a_custom_loss_function.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
