
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
# Đánh Giá Hiệu Năng GPT-2 Trên CPU và GPU: Thực Nghiệm Thời Gian Khởi Tạo, Suy Luận và Huấn Luyện

## Tóm tắt (Abstract)

Hiệu năng tính toán là yếu tố then chốt trong việc triển khai và huấn luyện các mô hình ngôn ngữ lớn. Bài viết này trình bày một nghiên cứu thực nghiệm nhằm so sánh thời gian thực thi mô hình GPT-2 (Model 5) trên CPU và GPU thông qua ba tác vụ chính: khởi tạo mô hình, suy luận (forward pass) và huấn luyện bằng lan truyền ngược (backpropagation). Kết quả cho thấy GPU mang lại lợi thế vượt trội về hiệu năng, đặc biệt trong các phép tính ma trận quy mô lớn, với mức cải thiện lên tới nhiều bậc độ lớn so với CPU.

---

## 1. Giới thiệu

Sự phát triển của các mô hình ngôn ngữ lớn (Large Language Models – LLMs) đã làm gia tăng nhu cầu về tài nguyên tính toán hiệu năng cao. Trong khi CPU phù hợp cho các tác vụ điều khiển và thử nghiệm ban đầu, GPU được tối ưu cho xử lý song song và các phép toán ma trận, vốn là nền tảng của học sâu.

Tài liệu tham khảo mô tả một bài toán thực hành nhằm đo lường thời gian thực thi của Model 5 trên CPU và GPU, tập trung vào ba giai đoạn: khởi tạo mô hình, suy luận và huấn luyện 

Mục tiêu của bài viết là:

* Đánh giá định lượng sự khác biệt hiệu năng giữa CPU và GPU.
* Phân tích nguyên nhân của sự chênh lệch.
* Thảo luận ý nghĩa thực tiễn đối với phát triển LLM.

---

## 2. Thiết lập thực nghiệm

### 2.1. Môi trường thực thi

Thực nghiệm được thực hiện trong môi trường có hỗ trợ GPU (ví dụ: NVIDIA A100), sử dụng thư viện PyTorch để xây dựng và triển khai mô hình.

Thiết bị được xác định thông qua biến `device`, cho phép tạo hai phiên bản mô hình:

* Một phiên bản trên CPU.
* Một phiên bản trên GPU.

Cách tiếp cận này giúp đảm bảo tính công bằng trong so sánh hiệu năng 

---

### 2.2. Điều chỉnh mã nguồn

Lớp mô hình được mở rộng thêm tham số `device` để đảm bảo các tensor được tạo đúng trên thiết bị tương ứng. Việc này nhằm tránh lỗi do tensor nằm trên CPU trong khi mô hình nằm trên GPU 

Ví dụ:

```python
self.device = device
tensor = torch.arange(..., device=self.device)

Cách thiết kế này giúp mã nguồn linh hoạt và ổn định hơn khi chuyển đổi giữa các thiết bị.

---

## 3. Thực nghiệm 1: Thời gian khởi tạo mô hình

### 3.1. Phương pháp

Trong thí nghiệm đầu tiên, thời gian được đo cho quá trình:

* Khởi tạo mô hình trên GPU.
* Khởi tạo mô hình trên CPU.

Không thực hiện forward pass hay huấn luyện, chỉ đánh giá chi phí tạo mô hình.

Quá trình được bao quanh bởi bộ đếm thời gian (clock timer) 

---

### 3.2. Kết quả

Kết quả điển hình:

* GPU: ~1.5 giây
* CPU: ~1.2 giây

Sự chênh lệch khoảng 300 ms là không đáng kể trong thực tế 

---

### 3.3. Phân tích

Do khởi tạo mô hình chỉ diễn ra một lần trong toàn bộ vòng đời hệ thống, nên sự khác biệt nhỏ này không ảnh hưởng nhiều đến hiệu suất tổng thể. Vì vậy, yếu tố quyết định không nằm ở giai đoạn khởi tạo.

---

## 4. Thực nghiệm 2: Đánh giá Forward Pass

### 4.1. Phương pháp

Trong thí nghiệm thứ hai, mô hình thực hiện suy luận trên dữ liệu giả:

* Batch size: 8
* Sequence length: 1024

Quy trình gồm:

1. Sinh tensor token ngẫu nhiên.
2. Chuyển sang thiết bị tương ứng.
3. Thực hiện forward pass.
4. Lặp lại 5 lần.

Trước khi đo thời gian, GPU được đồng bộ với CPU để đảm bảo độ chính xác 

---

### 4.2. Kết quả

Kết quả thực nghiệm:

* CPU: ~20 giây
* GPU: ~0.03 giây (30 ms)

GPU nhanh hơn CPU khoảng 4 bậc độ lớn 

---

### 4.3. Phân tích

Sự khác biệt lớn xuất phát từ:

* Khả năng xử lý song song của GPU.
* Tối ưu hóa phần cứng cho phép nhân ma trận.
* Băng thông bộ nhớ cao.

Trong bối cảnh sinh token liên tục, việc chờ 20 giây cho mỗi lượt suy luận là không khả thi, khiến CPU không phù hợp cho các hệ thống LLM thực tế.

---

## 5. Thực nghiệm 3: Đánh giá Backpropagation

### 5.1. Phương pháp

Thí nghiệm thứ ba đo thời gian huấn luyện thông qua lan truyền ngược:

* Xây dựng hàm mất mát (loss function).
* Khởi tạo bộ tối ưu (optimizer).
* Thực hiện 5 vòng backpropagation.

Quy trình được thực hiện riêng cho CPU và GPU 

---

### 5.2. Kết quả

Kết quả quan sát:

* GPU: ~1.6 giây
* CPU: > 60 giây

Sự chênh lệch vượt quá một phút cho cùng khối lượng tính toán 

---

### 5.3. Phân tích

Backpropagation yêu cầu:

* Nhiều phép nhân ma trận.
* Tính gradient quy mô lớn.
* Cập nhật tham số liên tục.

Các tác vụ này được GPU xử lý hiệu quả hơn nhiều so với CPU. Khi quy mô mô hình tăng (GPT-2 Medium, Large), khoảng cách này tiếp tục mở rộng.

---

## 6. Thảo luận

### 6.1. Ý nghĩa đối với phát triển LLM

Kết quả cho thấy:

* CPU chỉ phù hợp cho học tập và thử nghiệm nhỏ.
* GPU là điều kiện cần cho huấn luyện và triển khai LLM.
* Hiệu năng ảnh hưởng trực tiếp đến khả năng mở rộng mô hình.

Ngay cả với GPT-2 Small, việc thiếu GPU khiến mô hình gần như không khả thi trong ứng dụng thực tế 

---

### 6.2. Khía cạnh kinh tế và chính sách

Tài liệu cũng nhấn mạnh rằng:

* GPU hiệu năng cao là tài nguyên chiến lược.
* Các quốc gia và tập đoàn lớn cần lượng lớn GPU để phát triển AI.
* Việc kiểm soát xuất khẩu GPU là một biện pháp quản lý rủi ro AI.

Điều này cho thấy mối liên hệ chặt chẽ giữa công nghệ, kinh tế và an ninh trong kỷ nguyên AI 

---

### 6.3. Hạn chế của nghiên cứu

Một số hạn chế bao gồm:

* Chỉ thử nghiệm trên GPT-2 Small.
* Dữ liệu đầu vào là dữ liệu giả.
* Chưa xét đến huấn luyện phân tán đa GPU.

Các nghiên cứu tiếp theo có thể mở rộng sang mô hình lớn hơn và môi trường phân tán.

---

## 7. Kết luận

Bài viết đã trình bày một nghiên cứu thực nghiệm về hiệu năng của GPT-2 trên CPU và GPU, tập trung vào ba giai đoạn chính: khởi tạo, suy luận và huấn luyện.

Các kết quả chính gồm:

* Khởi tạo mô hình: khác biệt không đáng kể.
* Forward pass: GPU nhanh hơn CPU ~10⁴ lần.
* Backpropagation: GPU nhanh hơn CPU hàng chục lần.

Những kết quả này khẳng định GPU là nền tảng không thể thiếu cho việc phát triển và ứng dụng mô hình ngôn ngữ lớn hiện đại.

---

## Tài liệu tham khảo

[1] CodeChallenge: Time Model 5 on CPU and GPU, Lecture Transcript. 

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
| [Multi-Head Attention: Cơ Sở Lý Thuyết và Triển Khai Thực Tiễn](aero_llm_019_multihead_attention_theory_and_implementation.md) | [Xem bài viết →](aero_llm_019_multihead_attention_theory_and_implementation.md) |
| [aero llm 01 intro](aero_llm_01_intro.md) | [Xem bài viết →](aero_llm_01_intro.md) |
| [Tối Ưu Hóa Huấn Luyện Mô Hình Học Sâu Bằng GPU: Nguyên Lý và Thực Hành](aero_llm_020_working_on_the_gpu.md) | [Xem bài viết →](aero_llm_020_working_on_the_gpu.md) |
| [Triển Khai Mô Hình GPT-2 Hoàn Chỉnh Trên GPU: Kiến Trúc, Tối Ưu Hóa và Đánh Giá Hiệu Năng](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) | [Xem bài viết →](aero_llm_021_mo_hinh_gpt_2_hoan_chinh_tren_gpu.md) |
| 📌 **[Đánh Giá Hiệu Năng GPT-2 Trên CPU và GPU: Thực Nghiệm Thời Gian Khởi Tạo, Suy Luận và Huấn Luyện](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md)** | [Xem bài viết →](aero_llm_022_anh_gia_hieu_nang_gpt_2_tren_cpu_va_gpu.md) |
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
