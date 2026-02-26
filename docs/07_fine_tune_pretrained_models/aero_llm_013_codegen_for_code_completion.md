
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
# Mô Hình CodeGen Cho Bài Toán Hoàn Thành Mã Nguồn: Kiến Trúc, Huấn Luyện và Ứng Dụng

## Tóm tắt

Bài viết này phân tích mô hình **CodeGen** phục vụ nhiệm vụ hoàn thành mã nguồn (code completion), do **Salesforce** phát triển. Dựa trên tài liệu hướng dẫn thực nghiệm , nghiên cứu trình bày kiến trúc mô hình, cơ chế sinh mã, đặc điểm tokenizer, cũng như quy trình fine-tuning trên dữ liệu lập trình. Các công thức toán học được sử dụng nhằm làm rõ nguyên lý hoạt động của mô hình ngôn ngữ tự hồi quy trong sinh mã. Kết quả cho thấy CodeGen có khả năng sinh mã hợp lệ ở mức cơ bản, tuy nhiên chất lượng phụ thuộc mạnh vào quy mô mô hình và dữ liệu huấn luyện.

---

## 1. Giới thiệu

Trong lĩnh vực trí tuệ nhân tạo cho lập trình (AI for Code), các mô hình ngôn ngữ chuyên biệt ngày càng được sử dụng rộng rãi để:

* Gợi ý đoạn mã,
* Hoàn thành hàm,
* Sinh chương trình tự động,
* Hỗ trợ học lập trình.

Theo tài liệu , CodeGen là một trong những mô hình tiêu biểu, được phát hành với nhiều quy mô khác nhau, từ 350 triệu đến 16 tỷ tham số. Mô hình có thể tải trực tiếp từ nền tảng **Hugging Face**.

Mục tiêu nghiên cứu:

* Phân tích kiến trúc CodeGen,
* Mô tả cơ chế sinh mã nguồn,
* Đánh giá vai trò của quy mô mô hình,
* Làm rõ quy trình fine-tuning trên dữ liệu lập trình.

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ tự hồi quy

Cho chuỗi token mã nguồn:

$$

X=(x_1,x_2,\dots,x_n)

$$

Xác suất sinh chuỗi:

$$

P(X)=\prod_{i=1}^{n}P(x_i\mid x_1,\dots,x_{i-1};\theta)

$$

Trong đó:

* $x_i$: token thứ $i$,
* $\theta$: tham số mô hình.

Nhiệm vụ hoàn thành mã là ước lượng:

$$

x_{n+1}=\arg\max_x P(x\mid X)

$$

---

### 2.2. Hàm mất mát huấn luyện

Hàm cross-entropy:

$$

\mathcal{L}(\theta)=
-\frac{1}{N}\sum_{i=1}^{N}
\log P(y_i\mid x_i;\theta)

$$

Mục tiêu:

$$

\theta^*=\arg\min_\theta \mathcal{L}(\theta)

$$

---

### 2.3. Self-Attention trong Transformer

Cho đầu vào $X\in\mathbb{R}^{n\times d}$:

$$

Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V

$$

$$

\text{Attn}(Q,K,V)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

$$

Cơ chế này cho phép mô hình học quan hệ giữa các dòng lệnh trong chương trình.

---

## 3. Kiến trúc mô hình CodeGen

### 3.1. Cấu trúc tổng thể

Theo tài liệu , phiên bản CodeGen-350M có:

* 20 khối Transformer,
* Kích thước embedding: $d=1024$,
* Từ vựng: khoảng 50.000 token,
* Không có position embedding riêng biệt.

Cấu trúc mỗi block:

$$

\text{LN} \rightarrow \text{Attention} \rightarrow \text{MLP}

$$

---

### 3.2. Ma trận QKV hợp nhất

CodeGen sử dụng ma trận QKV ghép:

$$

W_{QKV}\in\mathbb{R}^{d\times 3d}

$$

Thay vì ba ma trận riêng:

$$

W_Q,W_K,W_V\in\mathbb{R}^{d\times d}

$$

Cách làm này giúp tối ưu tốc độ tính toán.

---

### 3.3. Mạng MLP mở rộng 4×

Lớp feed-forward:

$$

h' = W_2\sigma(W_1 h)

$$

với:

$$

W_1\in\mathbb{R}^{d\times 4d},\quad
W_2\in\mathbb{R}^{4d\times d}

$$

---

### 3.4. Đặc điểm embedding

Số hàng embedding:

$$

N_{emb}=51,200

$$

Trong khi số token:

$$

N_{tok}\approx 50,257

$$

Do đó tồn tại các vector “trống”:

$$

N_{emb}>N_{tok}

$$

nhằm tối ưu bộ nhớ GPU .

---

## 4. Tokenizer và xử lý dữ liệu

### 4.1. Tokenizer

Tokenizer của CodeGen được phát triển dựa trên tokenizer của **OpenAI** (GPT-2), có điều chỉnh cho mã nguồn.

Ký hiệu:

$$

V={w_1,\dots,w_{|V|}}

$$

là tập token.

---

### 4.2. Độ trùng lặp trong mã nguồn

Theo tài liệu :

* Tổng token: (160,000),
* Token duy nhất: (3,000).

Tỷ lệ đa dạng:

$$

r=\frac{3000}{160000}\approx1.9%

$$

Cho thấy mã nguồn có mức lặp cao.

---

### 4.3. Trích xuất dữ liệu từ GitHub

Dữ liệu được thu thập từ các kho trên **GitHub**, tập trung vào file `.ipynb`.

Tập dữ liệu:

$$

\mathcal{D}={x_1,\dots,x_N}

$$

với mỗi $x_i$ là một cell code.

---

## 5. Cơ chế sinh mã nguồn

### 5.1. Sinh token tuần tự

Với prompt ban đầu:

$$

X_0=(x_1,\dots,x_k)

$$

Mô hình sinh:

$$

x_{k+1}\sim P(x\mid X_0)

$$

Lặp lại:

$$

X_{t+1}=X_t\oplus x_{t+1}

$$

---

### 5.2. Temperature Sampling

Phân phối xác suất:

$$

p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}

$$

Trong đó:

* $z_i$: logit,

* $T$: temperature.

* $T\downarrow$: mã ổn định,

* $T\uparrow$: mã đa dạng.

---

### 5.3. Đánh giá tính hợp lệ

Gọi:

$$

f(x)=
\begin{cases}
1, & x\ \text{chạy được} \
0, & \text{lỗi}
\end{cases}

$$

Tỷ lệ hợp lệ:

$$

R=\frac{1}{M}\sum_{i=1}^{M}f(x_i)

$$

Với mô hình nhỏ:

$$

R_{350M}<R_{16B}

$$

.

---

## 6. Fine-tuning cho miền chuyên biệt

### 6.1. Mô hình fine-tuning

Tham số chia thành:

$$

\theta=(\theta_0,\Delta\theta)

$$

Trong đó:

* $\theta_0$: tiền huấn luyện,
* $\Delta\theta$: tham số cập nhật.

---

### 6.2. Huấn luyện trên mã giải tích

Dữ liệu từ sách giải tích được dùng để fine-tune, giúp mô hình sinh mã:

* Tích phân,
* Đồ thị,
* Hàm từng phần.

Hàm mục tiêu:

$$

\min_{\Delta\theta}
\mathcal{L}(\theta_0+\Delta\theta)

$$

---

### 6.3. Tác động của fine-tuning

Sau fine-tuning:

$$

P_{domain}(x)\approx P_{data}(x)

$$

⇒ mã sinh ra phù hợp miền dữ liệu.

---

## 7. Phân tích hiệu quả

### 7.1. Ảnh hưởng của quy mô mô hình

Gọi:

$$

P=\text{số tham số}

$$

Chất lượng trung bình:

$$

Q\propto\log(P)

$$

Mô hình lớn sinh mã hợp lệ tốt hơn.

---

### 7.2. Đánh đổi chi phí – hiệu năng

Giả sử:

$$

C\propto P

$$

Hiệu quả:

$$

E=\frac{Q}{C}

$$

Mô hình nhỏ có $E$ cao cho học tập, mô hình lớn phù hợp triển khai.

---

### 7.3. Hạn chế

Theo tài liệu :

* Mã sinh có thể không chạy được,
* Thiếu logic toàn cục,
* Dễ sinh nhiễu đa ngôn ngữ.

---

## 8. Ứng dụng thực tiễn

CodeGen được sử dụng trong:

* IDE gợi ý mã,
* Hỗ trợ học lập trình,
* Sinh script khoa học,
* Phân tích dữ liệu.

Đặc biệt phù hợp cho:

$$

N_{data}\ \text{nhỏ},\quad P\ \text{trung bình}

$$

---

## 9. Kết luận

Bài viết đã phân tích mô hình CodeGen cho bài toán hoàn thành mã nguồn dựa trên tài liệu thực nghiệm. Các kết luận chính:

1. CodeGen sử dụng kiến trúc Transformer chuyên cho mã.
2. Quy mô mô hình ảnh hưởng mạnh đến chất lượng.
3. Fine-tuning giúp thích nghi miền chuyên biệt.
4. Mô hình nhỏ phù hợp nghiên cứu, mô hình lớn phù hợp triển khai.

Trong tương lai, việc kết hợp CodeGen với PEFT và RLHF có thể nâng cao độ tin cậy của mã sinh tự động.

---

## Tài liệu tham khảo

1. Giới thiệu CodeGen cho Code Completion 
2. Vaswani et al. (2017). Attention Is All You Need.
3. Chen et al. (2021). Evaluating Large Language Models for Code.
4. Nijkamp et al. (2022). CodeGen: An Open Large Language Model for Code.
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
| 📌 **[Mô Hình CodeGen Cho Bài Toán Hoàn Thành Mã Nguồn: Kiến Trúc, Huấn Luyện và Ứng Dụng](aero_llm_013_codegen_for_code_completion.md)** | [Xem bài viết →](aero_llm_013_codegen_for_code_completion.md) |
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
