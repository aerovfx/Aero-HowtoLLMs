
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
Dưới đây là bài viết khoa học bằng tiếng Việt, được trình bày theo định dạng Markdown, dựa trên tài liệu bạn cung cấp , có bổ sung công thức toán học và tham khảo học thuật.

---

# Đánh Giá Định Lượng và Định Tính Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp Văn Phong *Alice* và *Edgar Allan Poe*

## Tóm tắt

Bài viết này trình bày phương pháp đánh giá hiệu quả của các mô hình ngôn ngữ sau quá trình fine-tuning dựa trên hai tập văn bản: *Alice's Adventures in Wonderland* và các tác phẩm của *Edgar Allan Poe*. Phương pháp đánh giá kết hợp hai hướng tiếp cận: định lượng (quantitative evaluation) dựa trên tần suất token, và định tính (qualitative evaluation) thông qua sinh văn bản. Kết quả cho thấy fine-tuning giúp mô hình học được đặc trưng phong cách, tuy nhiên việc lượng hóa chất lượng sinh văn bản vẫn còn nhiều hạn chế.

---

## 1. Giới thiệu

Các mô hình ngôn ngữ lớn (Large Language Models – LLMs) ngày càng được ứng dụng rộng rãi trong sinh văn bản. Tuy nhiên, việc đánh giá chất lượng đầu ra của các mô hình này vẫn là một thách thức lớn.

Trong nghiên cứu này, chúng tôi tập trung vào:

* Huấn luyện tinh chỉnh (fine-tuning) hai mô hình theo hai phong cách văn học khác nhau.
* Đánh giá mức độ “học phong cách” thông qua phân tích thống kê token.
* So sánh giữa đánh giá định lượng và đánh giá định tính.

Tài liệu thực nghiệm được trích dẫn từ bài hướng dẫn lập trình .

---

## 2. Cơ sở lý thuyết

### 2.1. Mô hình ngôn ngữ và Tokenization

Cho văn bản đầu vào:
$$
X = (x_1, x_2, \dots, x_n)
$$


Trong đó $x_i$ là các token sau khi mã hóa.

Mô hình học xác suất có điều kiện:
$$
P(X) = \prod_{i=1}^{n} P(x_i \mid x_1, \dots, x_{i-1})
$$


Quá trình tokenization chuyển văn bản sang dãy chỉ số số nguyên:
$$
T = (t_1, t_2, \dots, t_n)
$$


với $t_i \in \mathbb{N}$.

---

### 2.2. Fine-tuning mô hình

Fine-tuning là quá trình cập nhật tham số mô hình trên tập dữ liệu chuyên biệt.

Hàm mất mát phổ biến là Cross-Entropy:
$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_i)
$$


Trong đó:

* $x_i$: đầu vào,
* $y_i$: token mục tiêu,
* $N$: số mẫu huấn luyện.

Mục tiêu là:
$$
\min_{\theta} \mathcal{L}(\theta)
$$


với $\theta$ là tham số mô hình.

---

## 3. Phương pháp nghiên cứu

### 3.1. Dữ liệu

Hai tập văn bản:

* Văn bản *Alice*.
* Văn bản *Edgar*.

Sau tokenization, ta thu được hai tập token:
$$
T_A = (t_1^A, \dots, t_{n_A}^A), \quad
T_E = (t_1^E, \dots, t_{n_E}^E)
$$


---

### 3.2. Lọc token

Chỉ giữ lại token có độ dài ≥ 3 ký tự:
$$
T'_i =
\begin{cases}
t_i, & \text{nếu } |decode(t_i)| \ge 3 \
-1, & \text{ngược lại}
\end{cases}
$$


Các token có giá trị (-1) bị loại bỏ.

Mục tiêu: giảm nhiễu do dấu câu và ký tự đơn.

---

### 3.3. Xác định 100 token phổ biến nhất

Với tập token đã lọc (T'), ta tính tần suất:
$$
f(w) = \sum_{i=1}^{N} \mathbf{1}(T'_i = w)
$$


Trong đó:
$$
\mathbf{1}(x) =
\begin{cases}
1, & x = \text{đúng} \
0, & x = \text{sai}
\end{cases}
$$


Chọn 100 token có (f(w)) lớn nhất:
$$
S_{100} = {w_1, \dots, w_{100}}
$$


---

### 3.4. Đánh giá định lượng

Cho dãy token sinh ra:
$$
G = (g_1, g_2, \dots, g_M)
$$


Tỷ lệ token thuộc tập phổ biến:
$$
p = \frac{1}{M} \sum_{i=1}^{M} \mathbf{1}(g_i \in S_{100})
$$


Áp dụng cho:

* Mô hình Alice → tập Alice
* Mô hình Alice → tập Edgar
* Mô hình Edgar → tập Alice
* Mô hình Edgar → tập Edgar

Trước và sau fine-tuning.

---

### 3.5. Đánh giá định tính

Cung cấp cùng một prompt cho hai mô hình:

> *“What did the Red Queen say to Alice?”*

Sau đó so sánh:

* Ngữ điệu,
* Từ vựng,
* Mạch truyện,
* Sắc thái văn học.

Phương pháp này mang tính chủ quan nhưng phản ánh trải nghiệm người đọc.

---

## 4. Thực nghiệm

### 4.1. Sinh văn bản

Mỗi mô hình sinh:

* 10 lần lặp,
* Mỗi lần 100 token.

Tổng:
$$
M = 1000
$$


token cho mỗi mô hình.

Token đầu vào ngẫu nhiên được loại bỏ.

---

### 4.2. Tổ chức dữ liệu

Dữ liệu được biểu diễn dưới dạng ma trận:
$$
P =
\begin{bmatrix}
p_{AA} & p_{AE} \
p_{EA} & p_{EE}
\end{bmatrix}
$$


Trong đó:

* $p_{AA}$: Alice → Alice,
* $p_{AE}$: Alice → Edgar,
* $p_{EA}$: Edgar → Alice,
* $p_{EE}$: Edgar → Edgar.

Xét trước (pre) và sau (post) fine-tuning.

---

### 4.3. Kết quả

Trước fine-tuning:
$$
p_{AA} \approx p_{AE} \approx p_{EA} \approx p_{EE}
$$


Sau fine-tuning:
$$
p_{AA} > p_{AE}, \quad
p_{EE} > p_{EA}
$$


Hiện tượng này tạo thành “crossover interaction”, cho thấy mô hình đã học được đặc trưng văn phong.

---

## 5. Thảo luận

### 5.1. Ưu điểm

* Đơn giản, dễ triển khai.
* Không cần đánh giá thủ công.
* Phù hợp với phân tích quy mô lớn.

### 5.2. Hạn chế

1. Token phổ biến không mang nhiều đặc trưng phong cách.
2. Không phản ánh ngữ nghĩa sâu.
3. Không đánh giá được tính sáng tạo.
4. Nhạy cảm với nhiễu thống kê.

Ví dụ: các từ như *the, and, of* xuất hiện ở mọi thể loại.

---

### 5.3. Hướng cải tiến

Có thể mở rộng bằng:

* Độ đo perplexity:
$$
\text{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_i\right)
$$


* Embedding similarity:
$$
\cos(\theta) = \frac{u \cdot v}{|u||v|}
$$


* Đánh giá bằng LLM (LLM-as-Judge).
* Human evaluation có cấu trúc.

---

## 6. Kết luận

Nghiên cứu cho thấy phương pháp dựa trên token tần suất cao có thể phản ánh bước đầu hiệu quả fine-tuning. Tuy nhiên, nó chưa đủ để đánh giá toàn diện chất lượng sinh văn bản.

Việc kết hợp nhiều tiêu chí:

* Thống kê,
* Ngữ nghĩa,
* Đánh giá con người,

là hướng tiếp cận cần thiết trong tương lai.

---

## Tài liệu tham khảo

1. Tài liệu hướng dẫn fine-tuning và đánh giá mô hình 
2. Jurafsky, D., & Martin, J. (2023). *Speech and Language Processing*.
3. Vaswani et al. (2017). *Attention Is All You Need*.
4. OpenAI (2024). *Evaluating Large Language Models*.

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
| 📌 **[Đánh Giá Định Lượng và Định Tính Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp Văn Phong *Alice* và *Edgar Allan Poe*](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md)** | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tunin.md) |
| [Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) | [Xem bài viết →](aero_llm_07_codechallenge_quantify_the_aliceedgar_fine_tuning.md) |
| [Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) | [Xem bài viết →](aero_llm_08_codechallenge_a_chat_between_alice_and_edgar.md) |
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) | [Xem bài viết →](aero_llm_09_partial_fine_tuning_by_freezing_attention_weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
