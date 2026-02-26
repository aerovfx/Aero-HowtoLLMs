
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
# Lưu Trữ và Tải Lại Mô Hình Học Sâu Trong PyTorch và Hugging Face: Phương Pháp, Cấu Trúc và Đánh Giá

## Tóm tắt

Bài viết này trình bày các phương pháp lưu trữ và tải lại mô hình học sâu trong môi trường PyTorch và hệ sinh thái Hugging Face. Dựa trên tài liệu thực nghiệm , nghiên cứu phân tích cấu trúc dữ liệu mô hình, cơ chế tuần tự hóa (serialization), quy trình khôi phục tham số, và kỹ thuật đóng gói mô hình. Các công thức toán học được sử dụng nhằm mô hình hóa quá trình cập nhật và bảo toàn tham số. Kết quả cho thấy việc lưu – tải mô hình đúng cách đóng vai trò then chốt trong tái sử dụng, triển khai và nghiên cứu AI.

---

## 1. Giới thiệu

Trong quá trình huấn luyện mô hình học sâu, việc không lưu trữ kết quả sẽ dẫn đến mất toàn bộ tham số khi phiên làm việc kết thúc. Điều này đặc biệt quan trọng trong môi trường điện toán đám mây như **Google Colab**.

Theo tài liệu , tác giả trình bày cách lưu và tải lại mô hình ngôn ngữ GPT-2 bằng công cụ của **Hugging Face** và **PyTorch**.

Mô hình minh họa chính trong nghiên cứu là **GPT-2**, một mô hình ngôn ngữ tiền huấn luyện phổ biến.

Mục tiêu nghiên cứu:

* Phân tích cấu trúc dữ liệu mô hình,
* Mô tả cơ chế lưu – tải tham số,
* So sánh phương pháp Hugging Face và PyTorch,
* Đánh giá hiệu quả bảo toàn mô hình.

---

## 2. Cơ sở lý thuyết

### 2.1. Biểu diễn tham số mô hình

Một mô hình học sâu được đặc trưng bởi tập tham số:
$$
\theta = {W_1, W_2, \dots, W_L, b_1, b_2, \dots, b_L}
$$


Trong đó:

* $W_l$: ma trận trọng số,
* $b_l$: vector bias,
* $L$: số lớp.

Toàn bộ tập $\theta$ cần được lưu trữ để tái tạo mô hình.

---

### 2.2. Quá trình huấn luyện

Tham số được cập nhật theo gradient descent:
$$
\theta_{t+1}
============

\theta_t-\eta\nabla_\theta\mathcal{L}_t
$$


với:

* $\eta$: learning rate,
* $\mathcal{L}$: hàm mất mát.

Mục tiêu của việc lưu mô hình là bảo toàn $\theta_T$ tại thời điểm hội tụ.

---

### 2.3. State Dictionary

Trong PyTorch, trạng thái mô hình được biểu diễn bởi:
$$
\text{state_dict}={\theta_i}_{i=1}^{P}
$$


với $P$ là số tensor tham số.

---

## 3. Cấu trúc lưu trữ mô hình Hugging Face

### 3.1. Định dạng thư mục

Theo , mô hình Hugging Face không được lưu dưới dạng một file duy nhất mà là một thư mục gồm:

* `config.json`,
* `tokenizer.json`,
* `model.safetensors`,
* `version.txt`.

Cấu trúc:
$$
\mathcal{F}={f_1,f_2,\dots,f_k}
$$


Trong đó $f_k$ chứa toàn bộ tham số.

---

### 3.2. File trọng số

File `model.safetensors` chứa ma trận:
$$
W\in\mathbb{R}^{d\times d'}
$$


Dung lượng xấp xỉ:
$$
S\approx 4\times P \text{ bytes}
$$


với $P$ là số tham số dạng float32.

Ví dụ GPT-2 small:
$$
S\approx 474\text{ MB}
$$


.

---

### 3.3. Lệnh lưu mô hình

Phương thức:
$$
\text{model.save_pretrained(path)}
$$


Thực hiện ánh xạ:
$$
\theta \rightarrow \mathcal{F}_{path}
$$


---

## 4. Chỉnh sửa và kiểm chứng mô hình

### 4.1. Thao tác thay đổi embedding

Theo tài liệu, embedding được thay bằng vector 1:
$$
E_{ij}=1,\ \forall i,j
$$


Thay vì:
$$
E_{ij}\sim \mathcal{N}(0,\sigma^2)
$$


Điều này giúp kiểm tra tính đúng đắn khi tải lại mô hình.

---

### 4.2. So sánh tham số

Trước và sau khi chỉnh sửa:
$$
\Delta E = E_{new}-E_{old}
$$


Nếu:
$$
|\Delta E|_F>0
$$


⇒ mô hình đã thay đổi.

---

### 4.3. Khôi phục mô hình

Sử dụng:
$$
\text{from_pretrained(path)}
$$


Tái tạo:
$$
\theta_{load}\approx\theta_{save}
$$


---

## 5. Lưu trữ bằng PyTorch

### 5.1. Lưu state dictionary

Với PyTorch:
$$
\text{torch.save(state_dict, file.pt)}
$$


Biểu diễn:
$$
\theta \rightarrow file.pt
$$


Khác với Hugging Face, phương pháp này chỉ tạo một file.

---

### 5.2. Tải lại mô hình
$$
\theta \leftarrow \text{torch.load(file.pt)}
$$


và:
$$
\text{model.load_state_dict}(\theta)
$$


Giúp khôi phục tham số.

---

### 5.3. Tính toàn vẹn tham số

Sai số khôi phục:
$$
\varepsilon=|\theta_{load}-\theta_{orig}|_2
$$


Lý tưởng:
$$
\varepsilon\approx 0
$$


---

## 6. Đóng gói và di chuyển mô hình

### 6.1. Nén thư mục

Theo , sử dụng:
$$
\text{zip}(\mathcal{F})\rightarrow file.zip
$$


Tỷ lệ nén:
$$
r=\frac{S_{zip}}{S_{raw}}
$$


Thông thường:
$$
r\approx 0.8-0.9
$$


với mô hình lớn.

---

### 6.2. Giải nén
$$
file.zip \rightarrow \mathcal{F}'
$$


Sao cho:
$$
\mathcal{F}'\equiv\mathcal{F}
$$


---

### 6.3. Di chuyển môi trường

Quy trình:

1. Nén mô hình,
2. Tải về máy cá nhân,
3. Upload lên phiên mới,
4. Giải nén,
5. Load mô hình.

Đảm bảo:
$$
P(\text{lỗi})\approx 0
$$


---

## 7. Phương pháp đánh giá

### 7.1. So sánh đầu ra

Cho input $x$:
$$
y_{old}=f(x;\theta_{old})
$$

$$
y_{new}=f(x;\theta_{load})
$$


Sai lệch:
$$
\delta=|y_{old}-y_{new}|
$$


Nếu $\delta\approx0$ ⇒ khôi phục thành công.

---

### 7.2. Kiểm tra embedding

Trường hợp kiểm chứng bằng vector 1:
$$
E_{ij}=1 \Rightarrow \text{mean}(E)=1
$$


Nếu đúng ⇒ tải đúng mô hình.

---

### 7.3. Đánh giá độ ổn định

Tính phương sai đầu ra:
$$
\sigma^2=\frac{1}{N}\sum(y_i-\bar{y})^2
$$


Mô hình ổn định ⇒ $\sigma^2$ thấp.

---

## 8. Thảo luận

### 8.1. So sánh hai phương pháp

| Tiêu chí      | Hugging Face | PyTorch    |
| ------------- | ------------ | ---------- |
| Định dạng     | Thư mục      | File       |
| Dễ triển khai | Cao          | Trung bình |
| Linh hoạt     | Trung bình   | Cao        |
| Tính phổ quát | Thấp         | Cao        |

---

### 8.2. Ưu điểm

* Bảo toàn tri thức huấn luyện,
* Hỗ trợ tái sử dụng,
* Thuận tiện triển khai.

---

### 8.3. Hạn chế

* Dung lượng lớn,
* Phụ thuộc phiên bản,
* Khó chuẩn hóa liên thư viện.

---

## 9. Ứng dụng thực tiễn

Phương pháp lưu – tải mô hình được ứng dụng trong:

* Triển khai hệ thống NLP,
* Chia sẻ mô hình nghiên cứu,
* Fine-tuning nhiều giai đoạn,
* Học tập và giảng dạy AI.

Đặc biệt quan trọng trong môi trường cloud:
$$
T_{session}<T_{train}
$$


⇒ bắt buộc phải lưu mô hình.

---

## 10. Kết luận

Bài viết đã trình bày hệ thống các phương pháp lưu và tải mô hình trong PyTorch và Hugging Face. Các kết luận chính:

1. Hugging Face phù hợp triển khai nhanh,
2. PyTorch phù hợp tùy biến sâu,
3. Nén dữ liệu hỗ trợ di chuyển mô hình,
4. Kiểm chứng tham số là bước bắt buộc.

Trong tương lai, việc xây dựng chuẩn lưu trữ thống nhất cho mô hình AI là hướng nghiên cứu quan trọng.

---

## Tài liệu tham khảo

1. Saving and Loading Trained Models – Code Challenge 
2. Devlin et al. (2019). BERT.
3. Nijkamp et al. (2022). CodeGen.
4. Goodfellow et al. (2016). Deep Learning.
5. Paszke et al. (2019). PyTorch.

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
| 📌 **[Lưu Trữ và Tải Lại Mô Hình Học Sâu Trong PyTorch và Hugging Face: Phương Pháp, Cấu Trúc và Đánh Giá](aero_llm_020_saving_and_loading_trained_models.md)** | [Xem bài viết →](aero_llm_020_saving_and_loading_trained_models.md) |
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
