
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
# Phân Tích Hiệu Quả Fine-tuning và Targeted Freezing (Phần 2): Đánh Giá Bằng Trực Quan Hóa và Chuẩn Ma Trận

## Tóm tắt

Bài viết này tiếp tục nghiên cứu phương pháp fine-tuning kết hợp với đóng băng có mục tiêu (targeted freezing) trong mô hình ngôn ngữ lớn. Dựa trên kết quả trực quan hóa bằng biểu đồ loss, phân bố token, thời gian tính toán và chuẩn ma trận trọng số, nghiên cứu so sánh giữa mô hình huấn luyện toàn phần và mô hình đóng băng một phần. Kết quả cho thấy việc đóng băng có mục tiêu giúp giảm chi phí tính toán và tăng tính ổn định, tuy nhiên không phải lúc nào cũng đảm bảo hiệu quả sinh văn bản vượt trội.

---

## 1. Giới thiệu

Trong phần trước, chiến lược đóng băng có mục tiêu đã được trình bày như một phương pháp giảm chi phí fine-tuning. Phần tiếp theo của nghiên cứu tập trung vào:

* Trực quan hóa quá trình huấn luyện,
* So sánh động lực học (training dynamics),
* Đánh giá mức độ thay đổi trọng số,
* Phân tích thời gian tính toán.

Theo tài liệu , các biểu đồ trực quan đóng vai trò quan trọng trong việc hiểu rõ sự khác biệt giữa hai mô hình.

---

## 2. Cơ sở lý thuyết

### 2.1. Hàm mất mát trong mô hình sinh

Với tập dữ liệu:

$$
\mathcal{D}={(x_i,y_i)}_{i=1}^{N}
$$

Hàm mất mát cross-entropy:

$$
\mathcal{L}
 = 
-\frac{1}{N}\sum_{i=1}^{N}
\log P(y_i|x_i;\theta)
$$

Trong đó $\theta$ là tham số mô hình.

Mục tiêu huấn luyện:

$$
\theta^*=\arg\min_\theta \mathcal{L}
$$

---

### 2.2. Gradient Descent với tham số đóng băng

Quy tắc cập nhật:

$$
\theta_{t+1}
 = 
\theta_t-\eta\nabla_\theta\mathcal{L}
$$

Với tham số bị đóng băng:

$$
\nabla_{\theta_f}\mathcal{L}=0
$$

Suy ra:

$$
\theta_f^{(t+1)}=\theta_f^{(t)}
$$

---

### 2.3. Chuẩn ma trận trọng số

Cho ma trận trọng số attention:

$$
W_t\in\mathbb{R}^{m\times n}
$$

Hiệu tại bước $t$:

$$
\Delta W_t=W_t-W_{t-1}
$$

Chuẩn Frobenius:

$$
|\Delta W_t|_F
 = 
\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}(\Delta W_{ij})^2}
$$

Chuẩn này phản ánh mức độ thay đổi của mô hình theo thời gian.

---

## 3. Phương pháp nghiên cứu

### 3.1. Thiết lập thí nghiệm

Theo mô tả trong tài liệu , hai mô hình được huấn luyện song song:

* **Model A**: Huấn luyện toàn phần.
* **Model B**: Đóng băng phần lớn trọng số, chỉ fine-tuning một số lớp attention.

Hai mô hình có cùng:

$$
\theta_A^{(0)}=\theta_B^{(0)}
$$

và cùng thứ tự dữ liệu.

---

### 3.2. Trực quan hóa loss

Loss tại epoch $k$:

$$
\mathcal{L}_k^{(A)},\quad
\mathcal{L}_k^{(B)}
$$

Vẽ:

* Biểu đồ đường: $\mathcal{L}_k$ theo $k$,
* Biểu đồ scatter: $(\mathcal{L}_k^{(B$},\mathcal{L}_k^{$A$})).

Đường chuẩn:

$$
y=x
$$

dùng để đánh giá sự tương đồng.

---

### 3.3. Đánh giá phân bố token

Gọi:

* $S$: tập token phổ biến,
* $G=(g_1,\dots,g_M$): chuỗi sinh.

Tỷ lệ:

$$
p=\frac{1}{M}\sum_{i=1}^{M}\mathbf{1}(g_i\in S)
$$

So sánh trước và sau huấn luyện:

$$
\Delta p = p_{post}-p_{pre}
$$

---

### 3.4. Đo thời gian huấn luyện

Tổng thời gian:

$$
T=\sum_{k=1}^{K}t_k
$$

Tỷ lệ tiết kiệm:

$$
r=\frac{T_{train}-T_{freeze}}{T_{train}}
$$

---

## 4. Kết quả thực nghiệm

### 4.1. Phân tích hàm mất mát

Theo kết quả trong tài liệu :

* Mô hình train có độ dốc loss lớn,
* Mô hình freeze giảm chậm hơn.

Ví dụ:

$$
\mathcal{L}_{freeze}: 3.78 \rightarrow 2.65
$$

Trong khi:

$$
\mathcal{L}_{train}: \text{giảm mạnh hơn}
$$

Điều này cho thấy mô hình huấn luyện toàn phần học nhanh hơn.

---

### 4.2. Biểu đồ Scatter Loss

Các điểm dữ liệu nằm dưới đường $y=x$:

$$
\mathcal{L}^{(B)}>\mathcal{L}^{(A)}
$$

⇒ mô hình freeze thường có loss cao hơn.

Một số điểm trên đường chéo phản ánh giai đoạn đầu huấn luyện, khi hai mô hình còn tương tự nhau .

---

### 4.3. Phân bố token sinh

Kết quả cho thấy:

$$
\Delta p_A>0,\quad \Delta p_B>0
$$

Cả hai mô hình đều học được phong cách dữ liệu.

Tuy nhiên, trong một số lần thử:

$$
p_B>p_A
$$

Hiện tượng này được giải thích bởi tính ngẫu nhiên của sampling .

---

### 4.4. Chuẩn thay đổi trọng số

Quan sát:

$$
|\Delta W_t|_F
$$

* Lớn ở giai đoạn đầu,
* Giảm mạnh sau vài epoch,
* Tăng chậm về sau.

Mô hình freeze có:

$$
|\Delta W_t^{(B)}|_F
>
|\Delta W_t^{(A)}|_F
$$

cho thấy các lớp còn trainable phải “gánh” phần lớn quá trình học .

---

### 4.5. Thời gian tính toán

Theo tài liệu:

$$
T_{freeze}\approx 89s,\quad
T_{train}\approx 120s
$$

Tỷ lệ tiết kiệm:

$$
r\approx 25%
$$

Mặc dù không quá lớn, lợi ích sẽ tăng mạnh với mô hình lớn hơn.

---

## 5. Thí nghiệm đảo ngược chiến lược đóng băng

Trong bài tập 5, chiến lược được đảo ngược:

* Huấn luyện hầu hết mô hình,
* Đóng băng attention tầng cao.

Kết quả:

$$
\mathcal{L}_A \approx \mathcal{L}_B
$$

Các đường loss gần như trùng nhau .

Điều này cho thấy:

* Đóng băng một số lớp muộn ít ảnh hưởng tới hiệu năng tổng thể.

---

## 6. Thảo luận

### 6.1. Ý nghĩa của loss trong mô hình sinh

Trong mô hình phân loại:

$$
\min \mathcal{L}\Rightarrow \max \text{accuracy}
$$

Nhưng trong mô hình sinh:

$$
\min \mathcal{L} \not\Rightarrow \max \text{quality}
$$

Loss thấp không đảm bảo văn bản mạch lạc hay tự nhiên.

---

### 6.2. Tính ổn định huấn luyện

Mô hình freeze có:

$$
Var(\mathcal{L}_B)<Var(\mathcal{L}_A)
$$

⇒ ổn định hơn ở giai đoạn đầu.

---

### 6.3. Vai trò của interpretability

Theo tài liệu , việc chọn lớp đóng băng phụ thuộc nhiều vào nghiên cứu interpretability:

* Phân tích vai trò từng tầng,
* Hiểu cấu trúc tri thức nội tại,
* Xác định vùng cần fine-tune.

---

## 7. Ứng dụng thực tiễn

Phương pháp trong nghiên cứu phù hợp cho:

* Fine-tuning dữ liệu doanh nghiệp,
* NLP chuyên ngành,
* Hệ thống tài nguyên thấp,
* Huấn luyện nhanh mô hình thử nghiệm.

Đặc biệt hiệu quả khi:

$$
N_{data}\ll P_{model}
$$

(ví dụ: ít dữ liệu, nhiều tham số).

---

## 8. Kết luận

Bài viết đã phân tích chi tiết kết quả fine-tuning với targeted freezing thông qua trực quan hóa và chuẩn ma trận. Các kết luận chính:

1. Mô hình train học nhanh hơn nhưng kém ổn định.
2. Mô hình freeze tiết kiệm thời gian và ổn định hơn.
3. Chất lượng sinh văn bản không phụ thuộc hoàn toàn vào loss.
4. Chiến lược đóng băng cần thiết kế dựa trên interpretability.

Targeted freezing là một phương pháp đơn giản về mặt kỹ thuật nhưng phức tạp về mặt tối ưu.

---

## Tài liệu tham khảo

1. Fine-tuning and Targeted Freezing (Part 2) 
2. Vaswani et al. (2017). Attention Is All You Need.
3. Goodfellow et al. (2016). Deep Learning. MIT Press.
4. Hu et al. (2022). LoRA: Low-Rank Adaptation of LLMs.
5. Jurafsky & Martin (2023). Speech and Language Processing.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 07_fine_tune_pretrained_models](README.md) | [Xem bài viết →](README.md) |
| [Fine-tuning Có Mục Tiêu và Đóng Băng Chính Xác Trọng Số Trong Mô Hình Ngôn Ngữ Lớn](aero_llm_010_codechallenge_fine_tuning_and_targeted_freezing_part_1_.md) | [Xem bài viết →](aero_llm_010_codechallenge_fine_tuning_and_targeted_freezing_part_1_.md) |
| 📌 **[Phân Tích Hiệu Quả Fine-tuning và Targeted Freezing (Phần 2): Đánh Giá Bằng Trực Quan Hóa và Chuẩn Ma Trận](aero_llm_011_codechallenge_fine_tuning_and_targeted_freezing_part_2_.md)** | [Xem bài viết →](aero_llm_011_codechallenge_fine_tuning_and_targeted_freezing_part_2_.md) |
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
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
