
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [04 buildgpt](index.md)

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
Dưới đây là bài viết khoa học được tổng hợp từ tài liệu bạn cung cấp, có bổ sung trích dẫn và trình bày dưới dạng **Markdown**.

---

# Multi-Head Attention: Cơ Sở Lý Thuyết và Triển Khai Thực Tiễn

## Tóm tắt (Abstract)

Multi-Head Attention (MHA) là một thành phần cốt lõi trong kiến trúc Transformer, cho phép mô hình học đồng thời nhiều dạng quan hệ ngữ cảnh khác nhau trong chuỗi dữ liệu. Bài viết này trình bày cơ sở toán học, cơ chế hoạt động, cách triển khai và ý nghĩa thực nghiệm của multi-head attention dựa trên tài liệu học tập đi kèm. Qua đó, bài viết giúp làm rõ vai trò của việc phân tách không gian biểu diễn thành nhiều "đầu chú ý" (attention heads) nhằm nâng cao khả năng biểu diễn của mô hình.

---

## 1. Giới thiệu

Cơ chế Attention đã trở thành nền tảng của các mô hình xử lý ngôn ngữ tự nhiên hiện đại. Trong đó, multi-head attention mở rộng mô hình single-head attention bằng cách cho phép xử lý song song nhiều không gian đặc trưng.

Theo tài liệu tham khảo, multi-head attention được xây dựng bằng cách chia các ma trận attention thành nhiều ma trận con, giúp xử lý song song các vector token

Mục tiêu của bài viết là:

* Trình bày cách xây dựng multi-head attention.
* Phân tích cơ sở toán học.
* Giải thích lý do sử dụng nhiều head.
* Mô tả quy trình triển khai trong thực tế.

---

## 2. Cơ sở toán học của Attention

### 2.1. Ma trận Query, Key và Value

Trong attention, ba ma trận chính được xây dựng:

* Query $Q$
* Key $K$
* Value $V$

Chúng được tính như sau:

$$

Q = XW_Q,\quad K = XW_K,\quad V = XW_V

$$


Trong đó:

* $X$: Ma trận embedding.
* (W_Q, W_K, W_V): Ma trận trọng số huấn luyện.

Các chiều embedding được trộn lẫn thông qua phép nhân ma trận, không được giữ nguyên theo từng chiều ban đầu

---

### 2.2. Single-Head Attention

Với một head, attention được tính theo công thức:

$$

\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

$$


Trong đó $d_k$ là số chiều của vector key.

---

## 3. Cơ chế Multi-Head Attention

### 3.1. Phân tách thành nhiều Head

Multi-head attention chia các ma trận Q, K, V thành $H$ phần không chồng lấn:

$$

Q = [Q_1, Q_2, ..., Q_H]

$$


Mỗi head có kích thước:

$$

d_h = \frac{D}{H}

$$


với $D$ là số chiều embedding.

Việc chia này yêu cầu $D$ chia hết cho $H$

---

### 3.2. Attention trên từng Head

Với mỗi head $i$:

$$

\text{head}_i =
\text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_h}}\right)V_i

$$


Hệ số chuẩn hóa được điều chỉnh theo số chiều mới $D/H$

---

### 3.3. Kết hợp các Head

Sau khi tính attention cho từng head, kết quả được ghép nối:

$$

A = \text{Concat}(\text{head}_1,...,\text{head}_H)W_0

$$


Trong đó $W_0$ là ma trận tuyến tính dùng để trộn thông tin giữa các head.

Không sử dụng hàm phi tuyến tại bước này nhằm tránh làm mất thông tin học được từ từng head

---

## 4. Phân tích số lượng tham số

Một điểm quan trọng là multi-head attention **không làm tăng số lượng tham số huấn luyện** so với single-head attention.

Mặc dù số phép tính tăng lên, tổng số tham số vẫn giữ nguyên vì các ma trận trọng số không bị chia nhỏ từ đầu

---

## 5. Lý do sử dụng Multi-Head Attention

### 5.1. Học nhiều đặc trưng song song

Mỗi head có thể tập trung vào một dạng quan hệ khác nhau:

* Quan hệ cục bộ.
* Quan hệ dài hạn.
* Tương đồng ngữ nghĩa.
* Cấu trúc cú pháp.

Nhờ đó, mô hình có khả năng biểu diễn phong phú hơn

---

### 5.2. Góc nhìn thực nghiệm

Hiện nay, chưa có lý thuyết toán học hoàn chỉnh giải thích vì sao multi-head attention hiệu quả.

Theo tài liệu, lý do chính là:

> Các nhà phát triển thử nghiệm và nhận thấy mô hình hoạt động tốt hơn.
> Deep learning mang tính thực nghiệm cao.

---

## 6. Triển khai Multi-Head Attention

### 6.1. Cấu trúc lớp

Một lớp multi-head attention thường bao gồm:

* Số head: $H$
* Kích thước mỗi head: $d_h$
* Các ma trận: (W_Q, W_K, W_V, W_0)

Các ma trận này ban đầu có kích thước $D \times D$ và chỉ được chia trong quá trình forward pass

---

### 6.2. Quy trình Forward Pass

Quy trình cơ bản:

1. Tính Q, K, V từ embedding.
2. Reshape thành dạng:

$$

(B, T, H, d_h)

$$


3. Hoán vị chiều để phù hợp với hàm attention.
4. Tính attention song song.
5. Ghép các head.
6. Nhân với $W_0$.

Việc hoán vị chiều giúp tối ưu cho GPU, dù gây thêm chi phí xử lý

---

### 6.3. Theo dõi kích thước Tensor

Một số triển khai cho phép bật chế độ theo dõi kích thước tensor trong quá trình tính toán nhằm hỗ trợ debug và học tập

---

## 7. Ví dụ kích thước

Ví dụ với:

* Embedding: 128
* Số head: 4

Ta có:

$$

128 \rightarrow 4 \times 32 \rightarrow 128

$$


Trong quá trình tính toán, embedding được chia thành 4 head, mỗi head 32 chiều, sau đó ghép lại

---

## 8. Thảo luận

Multi-head attention mang lại các lợi ích chính:

* Tăng khả năng biểu diễn.
* Học đa dạng quan hệ.
* Cải thiện hiệu suất mô hình.
* Không làm tăng số tham số.

Tuy nhiên, chi phí tính toán và bộ nhớ cao hơn vẫn là một thách thức trong các mô hình quy mô lớn.

Ngoài ra, việc hiểu sâu cơ chế này hỗ trợ:

* Thiết kế kiến trúc mới.
* Tối ưu mô hình.
* Phân tích hành vi của LLM.

---

## 9. Kết luận

Bài viết đã trình bày:

* Cơ sở toán học của multi-head attention.
* Cách phân tách và kết hợp các head.
* Cơ chế triển khai trong thực tế.
* Lý do sử dụng nhiều head.

Multi-head attention là nền tảng quan trọng của các mô hình Transformer hiện đại, đóng vai trò quyết định trong sự thành công của các hệ thống ngôn ngữ lớn ngày nay.

---
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mở rộng Kiến trúc GPT: Position Embedding, Layer Normalization, Weight Tying và Temperature Scaling](aero_llm_010_posion_embedding.md) | [Xem bài viết →](aero_llm_010_posion_embedding.md) |
| [Biểu diễn Tính Nhân Quả Thời Gian trong Cơ Chế Attention bằng Đại Số Tuyến Tính](aero_llm_011_temporal_causality_via_linear_algebra_theory_.md) | [Xem bài viết →](aero_llm_011_temporal_causality_via_linear_algebra_theory_.md) |
| [Cơ Chế Trung Bình Hóa Quá Khứ và Loại Bỏ Tương Lai trong Mô Hình Ngôn Ngữ Nhân Quả](aero_llm_012_averaging_the_past_while_ignoring_the_future.md) | [Xem bài viết →](aero_llm_012_averaging_the_past_while_ignoring_the_future.md) |
| [Thuật Toán Attention trong Mô Hình Transformer: Cơ Sở Lý Thuyết, Cơ Chế Hoạt Động và Hàm Ý Ứng Dụng](aero_llm_013_the_attention_algorithm_theory_.md) | [Xem bài viết →](aero_llm_013_the_attention_algorithm_theory_.md) |
| [Phân Tích và Triển Khai Cơ Chế Attention: So Sánh Cài Đặt Thủ Công và PyTorch Tối Ưu](aero_llm_014_codechallenge_code_attention.md) | [Xem bài viết →](aero_llm_014_codechallenge_code_attention.md) |
| [Phân Tích Kiến Trúc Mô Hình Ngôn Ngữ với Một Attention Head: Lý Thuyết, Triển Khai và Đánh Giá](aero_llm_015_model.md) | [Xem bài viết →](aero_llm_015_model.md) |
| [Phân Tích Cấu Trúc Transformer Block: Lý Thuyết, Cơ Chế Biểu Diễn và Vai Trò Trong Mô Hình Ngôn Ngữ](aero_llm_016_the_transformer_block_theory_.md) | [Xem bài viết →](aero_llm_016_the_transformer_block_theory_.md) |
| [Cài Đặt Transformer Block Bằng PyTorch: Phân Tích Kiến Trúc, Luồng Dữ Liệu và Tối Ưu Hóa](aero_llm_017_the_transformer_block_code_.md) | [Xem bài viết →](aero_llm_017_the_transformer_block_code_.md) |
| [Mô Hình Nhiều Transformer Blocks Trong Mạng Ngôn Ngữ: Kiến Trúc, Phân Cấp Biểu Diễn và Khả Năng Mở Rộng](aero_llm_018_model_4_multiple_transformer_blocks_.md) | [Xem bài viết →](aero_llm_018_model_4_multiple_transformer_blocks_.md) |
| [aero llm 019 copy 10](aero_llm_019_copy_10.md) | [Xem bài viết →](aero_llm_019_copy_10.md) |
| [aero llm 019 copy 11](aero_llm_019_copy_11.md) | [Xem bài viết →](aero_llm_019_copy_11.md) |
| [aero llm 019 copy 12](aero_llm_019_copy_12.md) | [Xem bài viết →](aero_llm_019_copy_12.md) |
| [aero llm 019 copy 13](aero_llm_019_copy_13.md) | [Xem bài viết →](aero_llm_019_copy_13.md) |
| [aero llm 019 copy 9](aero_llm_019_copy_9.md) | [Xem bài viết →](aero_llm_019_copy_9.md) |
| 📌 **[Multi-Head Attention: Cơ Sở Lý Thuyết và Triển Khai Thực Tiễn](aero_llm_019_multihead_attention_theory_and_implementation.md)** | [Xem bài viết →](aero_llm_019_multihead_attention_theory_and_implementation.md) |
| [aero llm 01 intro](aero_llm_01_intro.md) | [Xem bài viết →](aero_llm_01_intro.md) |
| [Tối Ưu Hóa Huấn Luyện Mô Hình Học Sâu Bằng GPU: Nguyên Lý và Thực Hành](aero_llm_020_working_on_the_gpu.md) | [Xem bài viết →](aero_llm_020_working_on_the_gpu.md) |
| [Triển Khai Mô Hình GPT-2 Hoàn Chỉnh Trên GPU: Kiến Trúc, Tối Ưu Hóa và Đánh Giá Hiệu Năng](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) | [Xem bài viết →](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) |
| [Đánh Giá Hiệu Năng GPT-2 Trên CPU và GPU: Thực Nghiệm Thời Gian Khởi Tạo, Suy Luận và Huấn Luyện](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) | [Xem bài viết →](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) |
| [Khảo Sát Mô Hình GPT-2 Tiền Huấn Luyện của OpenAI: Kiến Trúc, Tham Số và Cơ Chế Sinh Văn Bản](aero_llm_023_inspecting_openai_s_gpt2.md) | [Xem bài viết →](aero_llm_023_inspecting_openai_s_gpt2.md) |
| [Kiến Trúc Transformer và Triển Khai GPT-2 trên GPU: Phân Tích Toán Học và Hiệu Năng Tính Toán](aero_llm_024_summarizing_gpt_using_equations.md) | [Xem bài viết →](aero_llm_024_summarizing_gpt_using_equations.md) |
| [Trực Quan Hóa Kiến Trúc GPT Thông Qua nano-GPT: Tiếp Cận Trực Quan trong Nghiên Cứu Mô Hình Ngôn Ngữ](aero_llm_025_visualizing_nano_gpt.md) | [Xem bài viết →](aero_llm_025_visualizing_nano_gpt.md) |
| [Phân Tích Số Lượng Tham Số Trong Mô Hình GPT-2: Phương Pháp Định Lượng và Ý Nghĩa Kiến Trúc](aero_llm_026_codechallenge_how_many_parameters_part_1_.md) | [Xem bài viết →](aero_llm_026_codechallenge_how_many_parameters_part_1_.md) |
| [Phân Bố Tham Số Trong GPT-2: So Sánh Attention, MLP và Layer Normalization](aero_llm_027_codechallenge_how_many_parameters_part_2_.md) | [Xem bài viết →](aero_llm_027_codechallenge_how_many_parameters_part_2_.md) |
| [📘 Phân Tích Kiến Trúc GPT-2: Từ Cơ Chế Multi-Head Attention Đến Hiệu Năng Tính Toán Trên GPU](aero_llm_028_codechallenge_gpt2_trained_weights_distributions.md) | [Xem bài viết →](aero_llm_028_codechallenge_gpt2_trained_weights_distributions.md) |
| [🧠 Phân Tích Nhân Quả Trong GPT-2: Vai Trò Của Ma Trận Query Thông Qua Can Thiệp Tham Số](aero_llm_029_codechallenge_do_we_really_need_q.md) | [Xem bài viết →](aero_llm_029_codechallenge_do_we_really_need_q.md) |
| [Phân Tích Kiến Trúc và Cơ Chế Hoạt Động của Mô Hình Ngôn Ngữ Transformer Cơ Bản](aero_llm_02_transformer.md) | [Xem bài viết →](aero_llm_02_transformer.md) |
| [Phân Tích Kỹ Thuật: So Sánh `nn.Embedding` và `nn.Linear` trong PyTorch](aero_llm_03_embedding_linear.md) | [Xem bài viết →](aero_llm_03_embedding_linear.md) |
| [Phân Tích So Sánh Hàm Kích Hoạt GELU và ReLU trong Mô Hình Ngôn Ngữ Lớn: Góc Nhìn Lý Thuyết và Thực Nghiệm](aero_llm_04_gelu_vs_relu_academic_analysis.md) | [Xem bài viết →](aero_llm_04_gelu_vs_relu_academic_analysis.md) |
| [Hàm Softmax và Tham Số Temperature trong Mô Hình Ngôn Ngữ Lớn: Phân Tích Toán Học và Thực Nghiệm](aero_llm_05_softmax_temperature_academic_analysis.md) | [Xem bài viết →](aero_llm_05_softmax_temperature_academic_analysis.md) |
| [Phân Tích `torch.multinomial`: Lấy Mẫu Xác Suất trong Sinh Văn Bản với PyTorch](aero_llm_06_torch_multinomial_academic_analysis.md) | [Xem bài viết →](aero_llm_06_torch_multinomial_academic_analysis.md) |
| [Phương Pháp Lấy Mẫu Token trong Sinh Văn Bản: Phân Tích So Sánh Greedy, Top-K, Top-P và Multinomial Sampling](aero_llm_07_token_sampling_methods.md) | [Xem bài viết →](aero_llm_07_token_sampling_methods.md) |
| [Phân Tích Hành Vi Của Hàm Softmax Trong Mô Hình Học Sâu: Ảnh Hưởng Của Lặp, Phạm Vi Số Học Và Nhiệt Độ](aero_llm_08_ham_softbank.md) | [Xem bài viết →](aero_llm_08_ham_softbank.md) |
| [Phân Tích Layer Normalization Trong Học Sâu: Cơ Sở Lý Thuyết, Ổn Định Số Học Và Ứng Dụng Thực Tiễn](aero_llm_09_layer_normalization.md) | [Xem bài viết →](aero_llm_09_layer_normalization.md) |
| [kien truc mo hinh ngon ngu lon](kien_truc_mo_hinh_ngon_ngu_lon.md) | [Xem bài viết →](kien_truc_mo_hinh_ngon_ngu_lon.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
