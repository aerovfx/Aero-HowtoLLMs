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
