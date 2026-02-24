# 🤖 Khóa học: Transformers & Large Language Models

> **Dựa trên giáo trình Stanford CME 295**
> *Biên soạn bởi Pixiboss*

Chào mừng bạn đến với bộ tài liệu hướng dẫn chuyên sâu về LLM. Tài liệu này đi từ những khái niệm nền tảng về kiến trúc Transformer cho đến các kỹ thuật huấn luyện, tinh chỉnh và xây dựng ứng dụng Agent hiện đại nhất.

## 📚 Mục Lục

### Phần 1: Nền Tảng Kiến Trúc (Foundations)
*   [**Bài 00: Tổng quan về LLM**](aero_LLM_00_Overview.md) - Bức tranh toàn cảnh về LLM và lịch sử phát triển.
*   [**Bài 01: Transformer Architecture**](aero_LLM_01_Transformer.md) - Trái tim của mọi mô hình ngôn ngữ hiện đại. Giải mã cơ chế Self-Attention.
*   [**Bài 02: Transformer Tricks & Optimizations**](aero_LLM_02_Transformer_Tricks.md) - Các kỹ thuật tối ưu hóa giúp Transformer hoạt động ổn định và hiệu quả hơn (Norm, Residual, Positional Encoding).
*   [**Bài 03: Giải mã các mô hình LLM (BERT, GPT, T5)**](aero_LLM_03_Large_Language_Models.md) - Phân loại các kiến trúc LLM phổ biến: Encoder-only, Decoder-only và Encoder-Decoder.

### Phần 2: Xây Dựng & Tinh Chỉnh (Building & Tuning)
*   [**Bài 04: Training & Pre-training**](aero_LLM_04_Training_Pretraining.md) - Quy trình huấn luyện mô hình từ con số 0. Scaling Laws và dữ liệu.
*   [**Bài 05: Fine-tuning & PEFT**](aero_LLM_05_Tuning_PEFT.md) - Tinh chỉnh mô hình hiệu quả với chi phí thấp bằng LoRA, QLoRA, Prompt Tuning.

> **💡 Góc kiến thức bổ trợ:** Để hiểu sâu về **RLHF** (Reinforcement Learning from Human Feedback) trong bài 5, bạn nên nắm vững các khái niệm cơ bản về RL:
> *   [**Reinforcement Learning Basics**](Reinforcement_Learning_Basics/README.md) (Bellman Equation, MDP, Policy vs Plan).

### Phần 3: Khả Năng Nâng Cao (Advanced Capabilities)
*   [**Bài 06: Reasoning & Prompt Engineering**](aero_LLM_06_Reasoning.md) - Kích hoạt khả năng suy luận của mô hình (Chain-of-Thought, Tree-of-Thought).
*   [**Bài 07: Agentic LLMs & Tool Use**](aero_LLM_07_Agentic_LLMs.md) - Biến LLM thành tác nhân tự chủ (Agent) biết sử dụng công cụ và RAG.

### Phần 4: Đánh Giá & Công Cụ (Evaluation & Tools)
*   [**Bài 08: Evaluation**](aero_LLM_08_Evaluation.md) - Làm sao để đo lường độ thông minh của AI? (Benchmarks, LLM-as-a-Judge).
*   [**Bài 09: Recap & Trends**](aero_LLM_09_Trends.md) - Tổng kết và nhìn về tương lai (Multimodal, Efficient AI).
*   [**Bài 10: Essential Tools for AI Engineers**](aero_LLM_10_Essential_Tools.md) 🆕 - Top 12 Repo quan trọng để tối ưu, chạy và tinh chỉnh LLM (vLLM, llama.cpp, Unsloth...).

---
*Tài liệu được lưu trữ tại `docs/01-01-LLM_Course` của repository Aero-HowtoLLMs.*
