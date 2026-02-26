
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
# Ứng Dụng Mô Hình BERT Trong Phân Loại Văn Bản Văn Học: Trường Hợp Alice và Edgar

## Tóm tắt

Bài viết này trình bày quá trình xây dựng và fine-tuning mô hình BERT nhằm phân loại các đoạn văn bản trích từ hai nguồn văn học: *Alice Through the Looking-Glass* của *Lewis Carroll* và tuyển tập thơ – truyện của *Edgar Allan Poe*. Dựa trên tài liệu thực nghiệm , nghiên cứu phân tích kiến trúc mô hình, quy trình huấn luyện, phương pháp làm mượt dữ liệu huấn luyện và đánh giá hiệu quả phân loại. Kết quả cho thấy BERT có khả năng phân biệt phong cách văn học với độ chính xác cao trong điều kiện dữ liệu huấn luyện hạn chế.

---

## 1. Giới thiệu

Trong xử lý ngôn ngữ tự nhiên, bài toán phân loại văn bản theo tác giả hoặc phong cách (author attribution) có ý nghĩa quan trọng trong nghiên cứu văn học số và đánh giá mô hình sinh văn bản.

Theo tài liệu , tác giả đã xây dựng một bộ phân loại dựa trên BERT để xác định xem một đoạn văn bản thuộc về “Alice” hay “Edgar”. Mô hình này còn được sử dụng làm công cụ đánh giá gián tiếp cho các mô hình sinh văn bản.

BERT do nhóm nghiên cứu tại **Google** phát triển, là nền tảng cho nhiều hệ thống NLP hiện đại.

Mục tiêu nghiên cứu:

* Phân tích mô hình BERT cho phân loại văn học,
* Mô tả quy trình fine-tuning,
* Trình bày phương pháp làm mượt (smoothing) dữ liệu huấn luyện,
* Đề xuất hướng ứng dụng trong đánh giá mô hình sinh.

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ hai chiều

BERT học biểu diễn ngữ cảnh hai chiều cho chuỗi token:

$$
X=(x_1,x_2,\dots,x_n)
$$

Biểu diễn ẩn tại vị trí $i$:

$$
h_i = f(x_1,\dots,x_n;\theta)
$$

Trong đó $\theta$ là tập tham số mô hình.

---

### 2.2. Biểu diễn [CLS] và phân loại

Với token đặc biệt [CLS], vector biểu diễn:

$$
h_{CLS}\in\mathbb{R}^d
$$

được dùng cho phân loại:

$$
z = W h_{CLS} + b
$$

$$
\hat{y}=\text{softmax}(z)
$$

Trong đó (W,b) là tham số của tầng phân loại.

---

### 2.3. Hàm mất mát Cross-Entropy

Với nhãn thật $y\in{0,1}$:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{2} y_{ic}\log(\hat{y}_{ic})
$$

Mục tiêu:

$$
\theta^{\ast}=\arg\min_\theta \mathcal{L}(\theta)
$$

---

## 3. Phương pháp nghiên cứu

### 3.1. Dữ liệu huấn luyện

Dữ liệu gồm các đoạn văn bản ngắn trích từ hai nguồn văn học khác nhau .

Tập dữ liệu:

$$
\mathcal{D}={(x_i,y_i)}_{i=1}^{N}
$$

Trong đó:

* $x_i$: chuỗi token,
* $y_i\in{0,1}$: nhãn Alice hoặc Edgar.

---

### 3.2. Thiết lập huấn luyện

Theo tài liệu :

* Batch size: 64,
* Độ dài chuỗi: 256 token,
* Số epoch: 150,
* Learning rate: rất nhỏ,
* Huấn luyện trên GPU.

Tổng số token:

$$
M = N\times L
$$

với $L=256$.

---

### 3.3. Quy trình fine-tuning

Quy trình gồm:

1. Token hóa dữ liệu,
2. Nạp mô hình BERT tiền huấn luyện,
3. Thêm tầng phân loại,
4. Huấn luyện bằng backpropagation,
5. Lưu mô hình sau huấn luyện.

Tham số được cập nhật theo:

$$
\theta_{t+1} = \theta_t-\eta\nabla_\theta\mathcal{L}_t
$$

---

### 3.4. Sinh batch và gán nhãn

Theo , mỗi batch gồm:

* 32 mẫu từ Alice,
* 32 mẫu từ Edgar.

Vector nhãn:

$$
y=(\underbrace{0,\dots,0}*{32}, \underbrace{1,\dots,1}*{32})
$$

---

## 4. Phương pháp làm mượt trung bình (Mean Smoothing)

### 4.1. Định nghĩa

Giả sử chuỗi loss:

$$
x=(x_1,x_2,\dots,x_n)
$$

Với cửa sổ kích thước $k$, giá trị làm mượt:

$$
y_i = \frac{1}{k} \sum_{j=i-w}^{i+w} x_j
$$

với:

$$
w=\frac{k-1}{2}
$$

---

### 4.2. Ý nghĩa

* Giảm nhiễu,
* Làm nổi bật xu hướng hội tụ,
* Hỗ trợ trực quan hóa.

Theo , giá trị $k=3$ cho kết quả cân bằng giữa mượt và trung thực.

---

### 4.3. Hiệu ứng biên

Tại biên chuỗi:

$$
i<w \quad \text{hoặc} \quad i>n-w
$$

sẽ xuất hiện sai lệch:

$$
y_i \approx \frac{1}{m}\sum x_j,\quad m<k
$$

Gây ra hiện tượng “edge effect”.

---

## 5. Phương pháp đánh giá

### 5.1. Độ chính xác (Accuracy)

$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N}\mathbf{1}(\hat{y}_i=y_i)
$$

Theo tài liệu , độ chính xác đạt mức cao chỉ sau vài chục epoch.

---

### 5.2. Hàm mất mát

Quá trình huấn luyện cho thấy:

$$
\mathcal{L}_{initial} > \mathcal{L}_{final}
$$

⇒ mô hình hội tụ.

---

### 5.3. Đánh giá định tính

Ngoài chỉ số định lượng, tác giả còn quan sát:

* Khả năng nhận diện phong cách,
* Độ ổn định dự đoán,
* Sự nhạy cảm với prompt.

---

## 6. Kết quả thực nghiệm

Theo kết quả trong :

* Accuracy tăng nhanh theo epoch,
* Loss giảm đều,
* Mô hình phân biệt tốt hai phong cách văn học.

Quan hệ giữa loss và epoch:

$$
\frac{d\mathcal{L}}{dt}<0
$$

Cho thấy xu hướng học ổn định.

Biểu đồ hai trục (loss–accuracy) giúp trực quan hóa quá trình hội tụ.

---

## 7. Thảo luận

### 7.1. Hiệu quả của learning rate nhỏ

Với $\eta$ nhỏ:

$$
|\theta_{t+1}-\theta_t|\ll1
$$

⇒ hạn chế phá vỡ tri thức tiền huấn luyện.

---

### 7.2. Vai trò trong đánh giá mô hình sinh

Mô hình phân loại có thể dùng để đo:

$$
S = P(\text{Alice} \mid \text{text})
$$

hoặc

$$
P(\text{Edgar} \mid \text{text})
$$

Từ đó đánh giá mô hình sinh văn bản theo phong cách.

---

### 7.3. Hạn chế

* Dữ liệu hạn chế,
* Chỉ phân loại nhị phân,
* Phụ thuộc chất lượng trích đoạn.

---

## 8. Ứng dụng thực tiễn

Phương pháp có thể áp dụng trong:

* Phân tích phong cách tác giả,
* Phát hiện đạo văn,
* Đánh giá mô hình sinh văn học,
* Hệ thống học tập văn chương số.

Trong nghiên cứu AI sáng tạo, bộ phân loại đóng vai trò “bộ giám khảo tự động”.

---

## 9. Kết luận

Bài viết đã trình bày quá trình fine-tuning BERT cho bài toán phân loại văn bản Alice – Edgar. Các kết quả chính:

1. BERT học nhanh phong cách văn học,
2. Learning rate nhỏ giúp ổn định mô hình,
3. Mean smoothing hỗ trợ phân tích huấn luyện,
4. Mô hình có tiềm năng làm công cụ đánh giá sinh văn bản.

Trong tương lai, có thể mở rộng sang phân loại đa tác giả và kết hợp với PEFT hoặc instruction tuning.

---

## Tài liệu tham khảo

1. BERT decides Alice or Edgar – Code Challenge 
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Jurafsky & Martin (2023). Speech and Language Processing.
4. Goodfellow et al. (2016). Deep Learning.

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
| 📌 **[Ứng Dụng Mô Hình BERT Trong Phân Loại Văn Bản Văn Học: Trường Hợp Alice và Edgar](aero_llm_021_bert_decides_alice_or_edgar.md)** | [Xem bài viết →](aero_llm_021_bert_decides_alice_or_edgar.md) |
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
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
