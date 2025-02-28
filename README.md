# Quy trình phát triển mô hình ngôn ngữ lớn LLMs

Kho lưu trữ này chứa mã nguồn để phát triển, tiền huấn luyện và tinh chỉnh một mô hình ngôn ngữ lớn (LLM) giống GPT và là kho mã chính thức cho cuốn sách [Quy trình phát triển mô hình ngôn ngữ lớn LLMs]

<br>
<br>

<br>

Trong [*Quy trình phát triển mô hình ngôn ngữ lớn LLMs*], bạn sẽ học và hiểu cách các mô hình ngôn ngữ lớn (LLMs) hoạt động từ bên trong bằng cách mã hóa chúng từ đầu, từng bước một. Trong cuốn sách này, tôi sẽ hướng dẫn bạn tạo ra LLM của riêng mình, giải thích từng giai đoạn với văn bản rõ ràng, sơ đồ và ví dụ.

Phương pháp được mô tả trong cuốn sách này để huấn luyện và phát triển mô hình nhỏ nhưng chức năng cho mục đích giáo dục phản ánh cách tiếp cận được sử dụng trong việc tạo ra các mô hình nền tảng quy mô lớn như những mô hình đằng sau ChatGPT. Ngoài ra, cuốn sách này còn bao gồm mã để tải trọng số của các mô hình đã được tiền huấn luyện lớn hơn để tinh chỉnh.

<<<<<<< HEAD
- Liên kết đến [kho mã nguồn chính thức](https://github.com/aerovfx/Aero-HowtoLLMs)

<br>
<br>

Để tải bản sao của kho lưu trữ này,thực hiện lệnh sau trong terminal của bạn:

```bash
git clone --depth 1 https://github.com/aerovfx/Aero-HowtoLLMs
```

<br>

# Mục lục

Xin lưu ý rằng tệp `README.md` này là tệp Markdown (`.md`). Nếu bạn đã tải gói mã này từ trang web của Manning và đang xem nó trên máy tính của mình, tôi khuyên bạn nên sử dụng trình soạn thảo hoặc trình xem trước Markdown để xem đúng cách. Nếu bạn chưa cài đặt trình soạn thảo Markdown, [MarkText](https://www.marktext.cc) là một lựa chọn miễn phí tốt.

<br>
<br>
<!--  -->

> [!TIP]
> Nếu bạn đang tìm kiếm hướng dẫn về cài đặt Python và các gói Python cũng như thiết lập môi trường mã, tôi đề xuất đọc tệp [README.md](setup/README.md) nằm trong thư mục [setup](setup).

<br>
<br>

[![Kiểm tra mã (Linux)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux.yml)
[![Kiểm tra mã (Windows)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows.yml)
[![Kiểm tra mã (macOS)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos.yml)

<br>

| Tiêu đề chương                                             | Mã chính (truy cập nhanh)                                                                                                       | Tất cả mã + bổ sung           |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [Khuyến nghị thiết lập](setup)                             | -                                                                                                                               | -                             |
| Ch 1: Hiểu về các mô hình ngôn ngữ lớn                     | Không có mã                                                                                                                     | -                             |
| Ch 2: Làm việc với dữ liệu văn bản                          | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (tóm tắt)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| Ch 3: Mã hóa cơ chế chú ý                                  | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (tóm tắt) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| Ch 4: Triển khai mô hình GPT từ đầu                        | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (tóm tắt)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| Ch 5: Tiền huấn luyện trên dữ liệu không gán nhãn          | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (tóm tắt) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (tóm tắt) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| Ch 6: Tinh chỉnh cho phân loại văn bản                     | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| Ch 7: Tinh chỉnh để làm theo hướng dẫn                     | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (tóm tắt)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (tóm tắt)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| Phụ lục A: Giới thiệu về PyTorch                            | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| Phụ lục B: Tài liệu tham khảo và đọc thêm                   | Không có mã                                                                                                                     | -                             |
| Phụ lục C: Giải pháp bài tập                                | Không có mã                                                                                                                     | -                             |
| Phụ lục D: Thêm các tính năng vào vòng lặp huấn luyện       | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| Phụ lục E: Tinh chỉnh hiệu quả với LoRA                     | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

Mô hình tinh thần dưới đây tóm tắt các nội dung được đề cập trong cuốn sách này.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
&nbsp;

## Yêu cầu phần cứng

Mã trong các chương chính của cuốn sách này được thiết kế để chạy trên các máy tính xách tay thông thường trong thời gian hợp lý và không yêu cầu phần cứng chuyên dụng. Cách tiếp cận này đảm bảo rằng một lượng lớn độc giả có thể tham gia vào tài liệu. Ngoài ra, mã tự động sử dụng GPU nếu có sẵn. (Vui lòng xem tài liệu [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) để biết thêm các khuyến nghị.)

&nbsp;
## Tài liệu bổ sung

Một số thư mục chứa tài liệu tùy chọn như một phần thưởng cho các độc giả quan tâm:

- **Thiết lập**
  - [Mẹo thiết lập Python](setup/01_optional-python-setup-preferences)
  - [Cài đặt các gói và thư viện Python được sử dụng trong cuốn sách này](setup/02_installing-python-libraries)
  - [Hướng dẫn thiết lập môi trường Docker](setup/03_optional-docker-environment)
- **Chương 2: Làm việc với dữ liệu văn bản**
  - [Mã hóa Byte Pair Encoding (BPE) từ đầu](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [So sánh các triển khai Byte Pair Encoding (BPE)](ch02/02_bonus_bytepair-encoder)
  - [Hiểu sự khác biệt giữa các lớp nhúng và các lớp tuyến tính](ch02/03_bonus_embedding-vs-matmul)
  - [Trực giác về Dataloader với các số đơn giản](ch02/04_bonus_dataloader-intuition)
- **Chương 3: Mã hóa cơ chế chú ý**
  - [So sánh các triển khai chú ý đa đầu hiệu quả](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [Hiểu về bộ đệm trong PyTorch](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **Chương 4: Triển khai mô hình GPT từ đầu**
  - [Phân tích FLOPS](ch04/02_performance-analysis/flops-analysis.ipynb)
- **Chương 5: Tiền huấn luyện trên dữ liệu không gán nhãn:**
  - [Tải trọng số thay thế từ Hugging Face Model Hub sử dụng Transformers](ch05/02_alternative_weight_loading/weight-loading-hf-transformers.ipynb)
  - [Tiền huấn luyện GPT trên tập dữ liệu Project Gutenberg](ch05/03_bonus_pretraining_on_gutenberg)
  - [Thêm các tính năng vào vòng lặp huấn luyện](ch05/04_learning_rate_schedulers)
  - [Tối ưu hóa siêu tham số cho tiền huấn luyện](ch05/05_bonus_hparam_tuning)
  - [Xây dựng giao diện người dùng để tương tác với LLM đã tiền huấn luyện](ch05/06_user_interface)
  - [Chuyển đổi GPT sang Llama](ch05/07_gpt_to_llama)
  - [Llama 3.2 từ đầu](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [Tải trọng số mô hình hiệu quả về bộ nhớ](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [Mở rộng bộ mã hóa BPE Tiktoken với các token mới](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
- **Chương 6: Tinh chỉnh cho phân loại**
  - [Các thí nghiệm bổ sung tinh chỉnh các lớp khác nhau và sử dụng các mô hình lớn hơn](ch06/02_bonus_additional-experiments)
  - [Tinh chỉnh các mô hình khác nhau trên tập dữ liệu đánh giá phim IMDB 50k](ch06/03_bonus_imdb-classification)
  - [Xây dựng giao diện người dùng để tương tác với bộ phân loại spam dựa trên GPT](ch06/04_user_interface)
- **Chương 7: Tinh chỉnh để làm theo hướng dẫn**
  - [Tiện ích tập dữ liệu để tìm các bản sao gần và tạo các mục giọng bị động](ch07/02_dataset-utilities)
  - [Đánh giá phản hồi hướng dẫn sử dụng API OpenAI và Ollama](ch07/03_model-evaluation)
  - [Tạo tập dữ liệu cho tinh chỉnh hướng dẫn](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [Cải thiện tập dữ liệu cho tinh chỉnh hướng dẫn](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Tạo tập dữ liệu ưu tiên với Llama 3.1 70B và Ollama](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [Tối ưu hóa ưu tiên trực tiếp (DPO) cho căn chỉnh LLM](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [Xây dựng giao diện người dùng để tương tác với mô hình GPT tinh chỉnh hướng dẫn](ch07/06_user_interface)

<br>
&nbsp;
=======
Kết luận
Việc xây dựng một mô hình ngôn ngữ lớn là một quá trình phức tạp, đòi hỏi sự kết hợp của nhiều công nghệ tiên tiến và nguồn lực tính toán lớn. Sự phát triển của LLMs mở ra nhiều cơ hội trong các lĩnh vực như trợ lý ảo, sáng tạo nội dung, dịch thuật, và nghiên cứu khoa học, nhưng cũng đặt ra nhiều thách thức về kiểm soát và đạo đức trong AI.

# LLMs in production
Overview of LLMs in Production

### AI Application + Data Products
- Q&A Webapp
- Chatbot
- Model as an API

### LLM Pipeline
- Corpus Creation
- Text Pre Processing
- Prompt Engineering
- LLM Inference
- Generated Text
### LLM Model(s)
- GPT 3.5
- GPT 4.0
- LLaMA
- Hugging Face
- MPT

>>>>>>> ab450f3d2e36437ab263df67da2a0772d02798e2
