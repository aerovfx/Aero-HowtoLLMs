
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [01 LLM Course](../../index.md) > [LectureStanford](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../index.md)
- [📚 Module 01: LLM Course](../../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# 🛠️ Top 12 Repo Quan Trọng Cho AI Engineer Tối Ưu LLM

> **Danh sách các công cụ "must-know" giúp tối ưu hóa, triển khai và tinh chỉnh LLM hiệu quả.**
> *Biên soạn bởi Pixiboss.*

---

## 🚀 1. Inference & Serving (Triển khai & Chạy mô hình)

### [vLLM](https://github.com/vllm-project/vllm)
*   **Đặc điểm:** Thư viện chạy inference LLM với tốc độ cực cao và thông lượng (throughput) lớn.
*   **Công nghệ:** Sử dụng **PagedAttention** để quản lý bộ nhớ KV cache hiệu quả.
*   **Ứng dụng:** Phù hợp nhất để triển khai hệ thống production quy mô lớn, phục vụ nhiều người dùng cùng lúc.

### [llama.cpp](https://github.com/ggerganov/llama.cpp)
*   **Đặc điểm:** Chạy LLM offline/local ngay trên máy tính cá nhân (MacBook, PC thường) mà không cần GPU quá mạnh.
*   **Công nghệ:** Tối ưu hóa tính toán trên CPU và Apple Silicon (Metal), sử dụng Quantization (GGUF) để giảm nhẹ mô hình.
*   **Ứng dụng:** Chạy LLM trên máy cấu hình yếu, thiết bị cá nhân.

### [Ollama](https://github.com/ollama/ollama)
*   **Đặc điểm:** Cách đơn giản nhất để chạy LLM local (như Llama 3, Mistral) chỉ với một câu lệnh (`ollama run llama3`).
*   **Công nghệ:** Bao bọc `llama.cpp` trong một giao diện thân thiện, dễ sử dụng.
*   **Ứng dụng:** Thử nghiệm nhanh mô hình, dev môi trường local.

### [MLC LLM](https://github.com/mlc-ai/mlc-llm)
*   **Đặc điểm:** Giải pháp triển khai LLM đa nền tảng (Universal Deployment).
*   **Ứng dụng:** Chạy LLM trên Mobile (iOS/Android), Web Browser (WebGPU), và các thiết bị Edge.

---

## 🧠 2. Framework & Core Libraries (Nền tảng cốt lõi)

### [Hugging Face Transformers](https://github.com/huggingface/transformers)
*   **Đặc điểm:** Thư viện tiêu chuẩn de-facto của cộng đồng AI. Cung cấp hàng ngàn mô hình pre-trained và pipeline sẵn sàng sử dụng.
*   **Ứng dụng:** Tải model, fine-tune, và xây dựng các ứng dụng NLP hiện đại.

### [PyTorch](https://github.com/pytorch/pytorch)
*   **Đặc điểm:** Framework Deep Learning phổ biến nhất thế giới nghiên cứu và sản phẩm AI hiện nay.
*   **Ứng dụng:** Là nền móng để xây dựng và huấn luyện hầu hết các mô hình AI hiện đại.

---

## ⚡ 3. Training & Fine-tuning (Huấn luyện & Tinh chỉnh)

### [Unsloth](https://github.com/unslothai/unsloth)
*   **Đặc điểm:** Thư viện Fine-tuning LLM (Llama-3, Mistral...) nhanh hơn 2x và tiết kiệm bộ nhớ VRAM hơn 60% so với cách thông thường.
*   **Công nghệ:** Viết lại các kernel tính toán đạo hàm (backpropagation) thủ công.
*   **Ứng dụng:** Fine-tune model trên GPU có VRAM hạn chế (như Colab miễn phí).

### [FlashAttention](https://github.com/Dao-AILab/flash-attention)
*   **Đặc điểm:** Thuật toán tăng tốc cơ chế Attention, giúp giảm bộ nhớ VRAM và chạy nhanh hơn.
*   **Ứng dụng:** Được tích hợp sâu vào PyTorch 2.0 và các thư viện khác để huấn luyện model với context window cực dài.

---

## 🌐 4. Distributed & System (Hệ thống phân tán)

### [exo](https://github.com/exo-explore/exo)
*   **Đặc điểm:** Biến các thiết bị rời rạc (MacBook, iPhone, iPad...) thành một "AI Cluster" tại nhà.
*   **Ứng dụng:** Chia tải (Inference Split) để chạy mô hình lớn trên nhiều thiết bị yếu kết hợp lại.

### [FastChat](https://github.com/lm-sys/FastChat)
*   **Đặc điểm:** Nền tảng mở để huấn luyện, phục vụ (serve) và đánh giá Chatbot.
*   **Sản phẩm nổi bật:** Là nền tảng đứng sau Vicuna và Chatbot Arena.
*   **Ứng dụng:** Xây dựng quy trình khép kín: Train -> Serve -> Eval.

---

## 🧪 5. Experimental & Deep Dive (Nghiên cứu sâu)

### [llm.c](https://github.com/karpathy/llm.c)
*   **Tác giả:** Andrej Karpathy.
*   **Đặc điểm:** Viết LLM (như GPT-2) bằng C và CUDA thuần túy, không dùng PyTorch/Python.
*   **Ứng dụng:** Tài liệu học tập tuyệt vời để hiểu sâu sắc cách mô hình hoạt động ở tầng thấp nhất (bare metal).

### [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
*   **Đặc điểm:** Phiên bản C++ tối ưu hóa của mô hình nhận dạng giọng nói Whisper (OpenAI).
*   **Ứng dụng:** Nhúng khả năng Speech-to-Text vào ứng dụng với tốc độ cực nhanh, không cần server.

---
*Biên soạn bởi Pixiboss.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠](aero_LLM_00_Overview.md) | [Xem bài viết →](aero_LLM_00_Overview.md) |
| [Lecture 1: Transformer Architecture 🤖](aero_LLM_01_Transformer.md) | [Xem bài viết →](aero_LLM_01_Transformer.md) |
| [Lecture 2: Transformer Tricks & BERT 🛠️](aero_LLM_02_Transformer_Tricks.md) | [Xem bài viết →](aero_LLM_02_Transformer_Tricks.md) |
| [Lecture 3: Large Language Models (LLMs) & Inference 🚀](aero_LLM_03_Large_Language_Models.md) | [Xem bài viết →](aero_LLM_03_Large_Language_Models.md) |
| [Lecture 4: LLM Training - Pre-training 🏋️](aero_LLM_04_Training_Pretraining.md) | [Xem bài viết →](aero_LLM_04_Training_Pretraining.md) |
| [Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️](aero_LLM_05_Tuning_PEFT.md) | [Xem bài viết →](aero_LLM_05_Tuning_PEFT.md) |
| [Lecture 6: LLM Reasoning 🧠](aero_LLM_06_Reasoning.md) | [Xem bài viết →](aero_LLM_06_Reasoning.md) |
| [Lecture 7: Agentic LLMs & Tool Use 🛠️](aero_LLM_07_Agentic_LLMs.md) | [Xem bài viết →](aero_LLM_07_Agentic_LLMs.md) |
| [Lecture 8: LLM Evaluation ⚖️](aero_LLM_08_Evaluation.md) | [Xem bài viết →](aero_LLM_08_Evaluation.md) |
| [Lecture 9: Recap & Current Trends 🔮](aero_LLM_09_Trends.md) | [Xem bài viết →](aero_LLM_09_Trends.md) |
| 📌 **[🛠️ Top 12 Repo Quan Trọng Cho AI Engineer Tối Ưu LLM](aero_LLM_10_Essential_Tools.md)** | [Xem bài viết →](aero_LLM_10_Essential_Tools.md) |
| [Chương 1: Tổng Quan Về Large Language Models (LLMs) 🧠](aero_LLM_chapter01_overview_detailed.md) | [Xem bài viết →](aero_LLM_chapter01_overview_detailed.md) |
| [Chương 2: 5 Trụ Cột Của Việc Huấn Luyện LLMs 🏛️](aero_LLM_chapter02_5pillars_part1.md) | [Xem bài viết →](aero_LLM_chapter02_5pillars_part1.md) |
| [Chương 2: 5 Trụ Cột - Part 2 (Evaluation & Systems)](aero_LLM_chapter02_5pillars_part2.md) | [Xem bài viết →](aero_LLM_chapter02_5pillars_part2.md) |
| [Chương 3: Pre-training → Post-training Pipeline 🔄](aero_LLM_chapter03_training_pipeline.md) | [Xem bài viết →](aero_LLM_chapter03_training_pipeline.md) |
| [Chương 4 & 5: Mechanisms & Evaluation 🔧📊](aero_LLM_chapter04_05_mechanisms_eval.md) | [Xem bài viết →](aero_LLM_chapter04_05_mechanisms_eval.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
