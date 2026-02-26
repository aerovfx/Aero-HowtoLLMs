
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
# Fine-tuning Mô hình GPT-2 trên Tác phẩm *Gulliver’s Travels*: Phân tích Thực nghiệm và Đánh giá Hiệu quả

## Tóm tắt (Abstract)

Bài viết này trình bày quy trình fine-tuning mô hình ngôn ngữ GPT-2 đã được huấn luyện sẵn trên văn bản *Gulliver’s Travels*. Thông qua việc phân tích tần suất token, đánh giá loss huấn luyện và chất lượng văn bản sinh ra, nghiên cứu cho thấy quá trình fine-tuning giúp mô hình thích nghi tốt hơn với phong cách văn bản mục tiêu. Tuy nhiên, việc giảm loss quá mức cũng tiềm ẩn nguy cơ overfitting, ảnh hưởng đến khả năng sáng tạo của mô hình.

---

## 1. Giới thiệu (Introduction)

Fine-tuning là một kỹ thuật phổ biến trong học sâu nhằm điều chỉnh mô hình đã được huấn luyện sẵn cho một nhiệm vụ hoặc tập dữ liệu cụ thể. Trong lĩnh vực xử lý ngôn ngữ tự nhiên, GPT-2 là một mô hình nền tảng mạnh mẽ, có thể được tinh chỉnh để thích nghi với phong cách văn bản chuyên biệt.

Theo tài liệu, mục tiêu chính của quá trình này là huấn luyện GPT-2 trên tác phẩm *Gulliver’s Travels* nhằm tạo ra văn bản mang phong cách tương tự, thay vì huấn luyện từ đầu. Cách tiếp cận này tận dụng tri thức ngôn ngữ đã học trước đó của mô hình .

---

## 2. Phương pháp (Methodology)

### 2.1. Mô hình và Môi trường Huấn luyện

Nghiên cứu sử dụng mô hình GPT-2 nền tảng được cung cấp bởi thư viện Hugging Face, kết hợp với PyTorch để huấn luyện. Mô hình được đưa lên GPU nhằm tăng tốc độ xử lý trong quá trình fine-tuning .

Các siêu tham số chính bao gồm:

* Batch size: 16
* Độ dài chuỗi: 256 token
* Learning rate: nhỏ hơn so với huấn luyện từ đầu
* Tối ưu hóa: Adam optimizer

Việc sử dụng learning rate nhỏ giúp tránh làm mất các tri thức ngôn ngữ đã được học từ trước .

---

### 2.2. Tiền xử lý Dữ liệu và Tokenization

Ban đầu, dữ liệu được mã hóa bằng phương thức `tokenizer.encode`, sau đó chuyển thành tensor PyTorch. Tuy nhiên, cách tiếp cận được tối ưu hóa bằng việc sử dụng trực tiếp:

```python

tokenizer(text, return_tensors="pt")

Phương pháp này trả về tensor trực tiếp, thuận tiện cho huấn luyện, nhưng tạo ra tensor hai chiều (1 × N). Do đó, cần truy cập hàng đầu tiên để chuyển về dạng một chiều .

---

### 2.3. Phân tích Tần suất Token

Để đánh giá mức độ “học phong cách” của mô hình, nghiên cứu xác định 100 token xuất hiện nhiều nhất trong *Gulliver’s Travels*. Các token phổ biến bao gồm dấu phẩy, xuống dòng, “the”, “and”,… .

Sau đó, mô hình được yêu cầu sinh văn bản và tính tỷ lệ token trong đầu ra thuộc nhóm 100 token phổ biến này.

---

### 2.4. Chiến lược Sinh Văn bản

Việc sinh văn bản sử dụng hàm `generate` với các tham số quan trọng:

* `do_sample=True`: đảm bảo tính ngẫu nhiên

* `bad_words_ids`: loại bỏ token kết thúc (EOS)

* `min_length = max_length`: đảm bảo độ dài cố định

Loại bỏ token EOS giúp mô hình không dừng sớm khi sinh văn bản, từ đó thu được đủ số lượng token cần thiết cho phân tích .

Ngoài ra, thay vì sinh một chuỗi dài 1000 token, mô hình sinh 10 chuỗi 100 token nhằm duy trì tính mạch lạc .

---

### 2.5. Huấn luyện và Tính Loss

Mô hình Hugging Face tích hợp sẵn hàm loss. Khi truyền tham số `labels` trùng với `input_ids`, mô hình tự động:

* Dịch chuỗi đầu vào sang phải 1 bước
* Áp dụng negative log-likelihood loss

Điều này giúp đơn giản hóa quá trình huấn luyện, không cần định nghĩa loss thủ công .

---

## 3. Kết quả Thực nghiệm (Experimental Results)

### 3.1. Tỷ lệ Token Đặc trưng

Trước khi fine-tuning, khoảng 40% token sinh ra thuộc nhóm token phổ biến trong *Gulliver’s Travels* .

Sau khi fine-tuning, con số này tăng lên khoảng 60% .

Điều này cho thấy mô hình đã thích nghi tốt hơn với phong cách văn bản mục tiêu.

---

### 3.2. Phân tích Văn bản Sinh ra

Sau fine-tuning, văn bản sinh ra có đặc điểm:

* Cấu trúc dòng ngắn
* Cách trình bày tương tự bản gốc
* Ngôn ngữ mang phong cách cổ điển

Ví dụ được cung cấp cho thấy nội dung sinh ra rất giống văn bản gốc về mặt ngữ điệu và bố cục .

---

### 3.3. Hành vi Loss

Loss huấn luyện giảm nhanh và tiến gần về 0 trong quá trình fine-tuning .

Mặc dù đây là dấu hiệu hội tụ tốt, nhưng cũng phản ánh nguy cơ mô hình ghi nhớ quá mức dữ liệu huấn luyện.

---

## 4. Thảo luận (Discussion)

### 4.1. Nguy cơ Overfitting

Loss tiến về 0 cho thấy mô hình có thể đã học thuộc văn bản huấn luyện, dẫn đến:

* Giảm khả năng tổng quát hóa
* Hạn chế tính sáng tạo
* Nguy cơ sao chép nội dung gốc

Tác giả nhấn mạnh rằng mục tiêu của mô hình sinh văn bản không phải là ghi nhớ hoàn toàn dữ liệu, mà là tạo ra nội dung mới, hợp lý và hữu ích .

---

### 4.2. Đánh giá Hiệu quả Fine-tuning

Việc sử dụng tỷ lệ token phổ biến là một chỉ số đơn giản nhưng hữu ích. Tuy nhiên, chỉ số này chủ yếu phản ánh đặc điểm bề mặt của ngôn ngữ, chưa đánh giá đầy đủ:

* Tính mạch lạc
* Tính sáng tạo
* Tính ngữ nghĩa

Do đó, cần kết hợp thêm các phương pháp đánh giá định tính và định lượng khác.

---

## 5. Kết luận (Conclusion)

Nghiên cứu cho thấy fine-tuning GPT-2 trên *Gulliver’s Travels* giúp mô hình:

* Gia tăng mức độ phù hợp phong cách
* Sinh văn bản gần với dữ liệu mục tiêu
* Giảm đáng kể loss huấn luyện

Tuy nhiên, việc giảm loss quá mạnh có thể dẫn đến overfitting. Do đó, trong các ứng dụng thực tế, cần cân nhắc giữa mức độ thích nghi và khả năng tổng quát hóa.

Các thách thức chính trong fine-tuning không nằm ở độ phức tạp của mã nguồn, mà ở việc:

* Lựa chọn dữ liệu phù hợp
* Điều chỉnh siêu tham số
* Thiết kế tiêu chí đánh giá hiệu quả .

---

## Tài liệu Tham khảo (References)

Tất cả các trích dẫn trong bài viết được lấy từ tài liệu:

* *2 - Fine-tune a pretrained GPT2.txt*
  (Nguồn nội bộ do người dùng cung cấp)

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
| 📌 **[Fine-tuning Mô hình GPT-2 trên Tác phẩm *Gulliver’s Travels*: Phân tích Thực nghiệm và Đánh giá Hiệu quả](aero_llm_02_fine_tune_a_pretrained_gpt2.md)** | [Xem bài viết →](aero_llm_02_fine_tune_a_pretrained_gpt2.md) |
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
