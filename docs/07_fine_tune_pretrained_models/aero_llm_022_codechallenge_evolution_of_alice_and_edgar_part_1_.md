
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
# Đồng Tiến Hóa Mô Hình Sinh Văn Bản và Mô Hình Phân Loại: Trường Hợp Alice và Edgar

## Tóm tắt

Bài viết này trình bày phương pháp kết hợp giữa mô hình sinh văn bản và mô hình phân loại nhằm đánh giá quá trình fine-tuning theo phong cách văn học. Dựa trên tài liệu thực nghiệm , nghiên cứu sử dụng ba mô hình trong cùng một môi trường: một bộ phân loại BERT và hai mô hình sinh văn bản được fine-tune theo phong cách Alice và Edgar. Bộ phân loại được dùng như một “giám khảo tự động” để đánh giá chất lượng sinh văn bản. Các công thức toán học được sử dụng nhằm mô hình hóa quá trình huấn luyện, chuyển đổi tokenizer và đánh giá hiệu năng.

---

## 1. Giới thiệu

Trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP), việc đánh giá chất lượng mô hình sinh văn bản theo phong cách (style transfer) vẫn là một thách thức. Phương pháp đánh giá thủ công tốn nhiều thời gian và thiếu tính khách quan.

Theo tài liệu , tác giả đề xuất một phương pháp thay thế: sử dụng mô hình phân loại dựa trên **BERT** để phân biệt văn bản do hai mô hình sinh tạo ra, từ đó đánh giá gián tiếp hiệu quả fine-tuning.

Hai phong cách văn học được lựa chọn dựa trên:

* *Alice's Adventures in Wonderland* – **Lewis Carroll**
* Tác phẩm của **Edgar Allan Poe**

Mục tiêu nghiên cứu:

* Phân tích mô hình kết hợp sinh – phân loại,
* Trình bày cơ chế đồng tiến hóa (co-evolution),
* Mô hình hóa toán học quá trình huấn luyện,
* Đánh giá vai trò của tokenizer và bộ nhớ GPU.

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ sinh tự hồi quy

Cho chuỗi token:
$$
X=(x_1,x_2,\dots,x_n)
$$


Xác suất sinh:
$$
P(X)=\prod_{i=1}^{n}P(x_i\mid x_{<i};\theta_g)
$$


Trong đó $\theta_g$ là tham số mô hình sinh.

---

### 2.2. Mô hình phân loại văn bản

Với đầu ra [CLS]:
$$
h_{CLS}\in\mathbb{R}^d
$$


Bộ phân loại:
$$
z = Wh_{CLS}+b
$$

$$
\hat{y}=\text{softmax}(z)
$$


Trong đó $\hat{y}$ là xác suất Alice/Edgar.

---

### 2.3. Hàm mất mát

#### $a$ Mô hình sinh
$$
\mathcal{L}_{gen}
=================

-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})
$$


#### $b$ Mô hình phân loại
$$
\mathcal{L}_{cls}
=================

-\frac{1}{N}\sum_{i=1}^{N}\sum_{c}y_{ic}\log\hat{y}_{ic}
$$


---

## 3. Phương pháp nghiên cứu

### 3.1. Kiến trúc hệ thống ba mô hình

Theo , hệ thống gồm:

1. BERT phân loại (đã fine-tune),
2. Mô hình sinh Alice,
3. Mô hình sinh Edgar.

Hai mô hình sinh dựa trên **EleutherAI** GPT-Neo:

> **GPT-Neo 125M**

Sơ đồ tổng quát:
$$
\text{Alice/Edgar} \rightarrow \text{Text} \rightarrow \text{BERT} \rightarrow \text{Label}
$$


---

### 3.2. Quản lý bộ nhớ và Half Precision

BERT được chuyển sang half precision:
$$
\text{float32} \rightarrow \text{float16}
$$


Giảm dung lượng:
$$
M_{fp16}\approx \frac{1}{2}M_{fp32}
$$


Giúp tiết kiệm GPU.

---

### 3.3. Dịch chuyển tokenizer

Hai tokenizer khác nhau:

* Tokenizer GPT-Neo,
* Tokenizer BERT.

Ánh xạ gián tiếp:
$$
T_{bert}(T^{-1}_{neo}(x))
$$


Trong đó:

* $T_{neo}$: encode GPT-Neo,
* $T_{bert}$: encode BERT.

Quy trình:
$$
\text{Token}*{neo}
\rightarrow \text{Text}
\rightarrow \text{Token}*{bert}
$$


---

### 3.4. Tạo batch huấn luyện

Theo tài liệu :

* Batch size: 64,
* Sequence length: 128,
* 32 Alice + 32 Edgar.

Ma trận batch:
$$
B\in\mathbb{R}^{64\times128}
$$


Vector nhãn:
$$
y=(\underbrace{0,\dots,0}*{32},
\underbrace{1,\dots,1}*{32})
$$


---

## 4. Chiến lược sinh dữ liệu

### 4.1. Sinh dư token

Để đảm bảo đủ token BERT:
$$
L_{neo}=kL_{bert},\quad k>1
$$


Trong thực nghiệm:
$$
k\approx4
$$


Sau đó cắt:
$$
X_{bert}=X_{neo}[1:L]
$$


---

### 4.2. Loại bỏ token không mong muốn

Danh sách token xấu:
$$
\mathcal{B}={\text{space},\text{tab},\text{newline},\dots}
$$


Ràng buộc sinh:
$$
x_t\notin\mathcal{B}
$$


---

### 4.3. Repetition Penalty

Hạn chế lặp:
$$
p_i'=\frac{p_i}{r^{c_i}}
$$


Trong đó:

* $c_i$: số lần lặp token,
* (r>1): hệ số phạt.

---

## 5. Phương pháp đánh giá

### 5.1. Độ chính xác phân loại
$$
\text{Acc}
==========

\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i=y_i)
$$


Trước fine-tuning:
$$
\text{Acc}\approx 0.5
$$


.

---

### 5.2. Hàm mất mát BERT
$$
\mathcal{L}*{cls}^{(t+1)}
<
\mathcal{L}*{cls}^{(t)}
$$


⇒ mô hình sinh tiến gần phong cách mục tiêu.

---

### 5.3. Đánh giá đồng tiến hóa

Gọi:
$$
S(t)=P_{BERT}(\text{Alice}\mid X_t)
$$


Nếu:
$$
S(t)\uparrow
$$


⇒ mô hình Alice cải thiện.

---

## 6. Kết quả thực nghiệm

Theo :

* Khi chưa fine-tune: Acc ≈ 45–50%,
* Sau fine-tune: Acc tăng dần,
* Loss giảm ổn định.

Quan hệ tổng quát:
$$
\frac{d}{dt}\mathcal{L}_{cls}<0
$$


Cho thấy quá trình hội tụ.

---

## 7. Thảo luận

### 7.1. Đồng tiến hóa sinh – phân loại

Hệ thống tạo vòng lặp:
$$
\text{Generate}\rightarrow\text{Classify}\rightarrow\text{Optimize}
$$


Giống mô hình học đối kháng nhẹ (weak adversarial learning).

---

### 7.2. Vai trò của tokenizer

Sai lệch tokenizer:
$$
|T_{neo}(x)|\ne|T_{bert}(x)|
$$


Là nguồn gây nhiễu chính trong huấn luyện.

---

### 7.3. Hạn chế

* Phụ thuộc mạnh vào BERT,
* Chi phí GPU lớn,
* Token translation phức tạp,
* Dễ nhiễu với dữ liệu nhỏ.

---

## 8. Ứng dụng thực tiễn

Phương pháp này có thể ứng dụng trong:

* Đánh giá mô hình sáng tạo văn học,
* Huấn luyện chatbot theo phong cách,
* Phát hiện đạo văn,
* Nghiên cứu AI sáng tạo.

Mô hình “giám khảo tự động” giúp giảm phụ thuộc vào con người.

---

## 9. Kết luận

Bài viết đã trình bày hệ thống kết hợp mô hình sinh và mô hình phân loại trong nghiên cứu Alice–Edgar. Các kết luận chính:

1. BERT có thể dùng làm bộ đánh giá phong cách,
2. Đồng tiến hóa giúp đo lường hiệu quả fine-tuning,
3. Tokenizer là yếu tố then chốt,
4. Half precision giúp tối ưu tài nguyên.

Trong tương lai, có thể mở rộng sang học tăng cường (RLHF) và đa phong cách.

---

## Tài liệu tham khảo

1. Evolution of Alice and Edgar (Part 1) – Code Challenge 
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Nijkamp et al. (2022). CodeGen: An Open Large Language Model for Code.
4. Vaswani et al. (2017). Attention Is All You Need.
5. Goodfellow et al. (2016). Deep Learning.

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
| 📌 **[Đồng Tiến Hóa Mô Hình Sinh Văn Bản và Mô Hình Phân Loại: Trường Hợp Alice và Edgar](aero_llm_022_codechallenge_evolution_of_alice_and_edgar_part_1_.md)** | [Xem bài viết →](aero_llm_022_codechallenge_evolution_of_alice_and_edgar_part_1_.md) |
| [📘 Đánh Giá Mô Hình Sinh Văn Bản Thông Qua Phân Loại BERT: Nghiên Cứu Trường Hợp Alice và Edgar](aero_llm_023_codechallenge_evolution_of_alice_and_edgar_part_2_.md) | [Xem bài viết →](aero_llm_023_codechallenge_evolution_of_alice_and_edgar_part_2_.md) |
| [Fine-tuning Mô hình GPT-2 trên Tác phẩm *Gulliver’s Travels*: Phân tích Thực nghiệm và Đánh giá Hiệu quả](aero_llm_02_fine_tune_a_pretrained_gpt2.md) | [Xem bài viết →](aero_llm_02_fine_tune_a_pretrained_gpt2.md) |
| [Đánh giá Ảnh hưởng của Learning Rate trong Fine-tuning GPT-2 trên *Gulliver’s Travels*](aero_llm_03codechallenge_gulliver_s_learning_rates.md) | [Xem bài viết →](aero_llm_03codechallenge_gulliver_s_learning_rates.md) |
| [Nghiên cứu Quy trình Sinh Văn bản từ Mô hình Ngôn ngữ Tiền Huấn luyện GPT-2](aero_llm_04_on_generating_text_from_pretrained_models.md) | [Xem bài viết →](aero_llm_04_on_generating_text_from_pretrained_models.md) |
| [Tinh Chỉnh Mô Hình GPT-2 Bằng Hàm Mất Mát KL Divergence Để Tối Ưu Hóa Việc Sinh Token Chứa Ký Tự “X”](aero_llm_05_codechallenge_maximize_the_x_factor_.md) | [Xem bài viết →](aero_llm_05_codechallenge_maximize_the_x_factor_.md) |
| [Tinh Chỉnh Mô Hình GPT-Neo Để Mô Phỏng Phong Cách Văn Học Alice in Wonderland và Edgar Allan Poe](aero_llm_06_alice_in_wonderland_and_edgar_allen_poe_with_gpt_neo_.md) | [Xem bài viết →](aero_llm_06_alice_in_wonderland_and_edgar_allen_poe_with_gpt_neo_.md) |
| [Đánh Giá Định Lượng và Định Tính Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp Văn Phong *Alice* và *Edgar Allan Poe*](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) |
| [Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) |
| [Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) | [Xem bài viết →](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) |
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
