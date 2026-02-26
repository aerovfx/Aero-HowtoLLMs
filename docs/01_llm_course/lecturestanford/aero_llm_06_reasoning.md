
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [01 llm course](../index.md) > [lecturestanford](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Lecture 6: LLM Reasoning 🧠

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này đi sâu vào khả năng suy luận của LLM, các mô hình Reasoning (như o1, R1) và kỹ thuật Reinforcement Learning (GRPO) để huấn luyện chúng.

---

## 📚 Mục Lục
1. [Reasoning là gì?](#1-reasoning-là-gì)
2. [Điểm yếu của Vanilla LLM](#2-điểm-yếu-của-vanilla-llm)
3. [Chain-of-Thought (CoT) & Inference-time Compute](#3-chain-of-thought-cot--inference-time-compute)
4. [Training Reasoning Models (Huấn luyện mô hình suy luận)](#4-training-reasoning-models)
5. [GRPO (Group Relative Policy Optimization)](#5-grpo-group-relative-policy-optimization)
6. [DeepSeek-R1 Pipeline](#6-deepseek-r1-pipeline)

---

## 1. Reasoning là gì?
**Reasoning (Suy luận)** là khả năng giải quyết các vấn đề phức tạp (như toán học, lập trình) thông qua một quy trình suy nghĩ nhiều bước (multi-step reasoning process).

*   *Câu hỏi kiến thức:* "Thủ đô của Pháp là gì?" -> Paris. (Không cần suy luận).

$$
*   *Câu hỏi suy luận:* "Một con gấu sinh năm 2020, năm 2025 nó bao nhiêu tuổi?" -> Cần tính toán: 2025 - 2020 = 5.
$$

---

## 2. Điểm yếu của Vanilla LLM
Các mô hình LLM tiêu chuẩn (như GPT-4 ban đầu, Llama 3) có một số hạn chế:
1.  **Limited Reasoning (Suy luận hạn chế):** Dễ bị "lạc lối" trong các bài toán nhiều bước phức tạp.
2.  **Static Knowledge (Kiến thức tĩnh):** Bị giới hạn bởi ngày cắt dữ liệu (knowledge cutoff).
3.  **No Action (Không hành động):** Chỉ nói (talk) chứ không làm (action).

---

## 3. Chain-of-Thought (CoT) & Inference-time Compute
Để giải quyết vấn đề suy luận, ta cần mô hình "nghĩ" (think) trước khi trả lời.

*   **Thinking Tokens:** Mô hình sinh ra một chuỗi suy luận (reasoning chain) nằm trong thẻ `<think>...</think>` trước khi đưa ra đáp án cuối cùng.
*   **Inference-time Compute:** Thay vì chỉ scale model size (Pre-training compute), ta tăng lượng tính toán tại thời điểm suy luận (cho mô hình nghĩ lâu hơn).
    *   *System 1 (Thinking fast):* Trả lời ngay lập tức (Vanilla LLM).
    *   *System 2 (Thinking slow):* Suy nghĩ kỹ rồi mới trả lời (Reasoning Models like o1, R1).

---

## 4. Training Reasoning Models
Làm sao dạy mô hình biết suy luận?

*   **SFT (Supervised Fine-Tuning):** Cần dữ liệu mẫu về quy trình suy luận (CoT data). *Khó khăn:* Dữ liệu suy luận chất lượng cao rất đắt và khan hiếm.
*   **RL (Reinforcement Learning):** Sử dụng các bài toán có đáp án kiểm chứng được (Verifiable Rewards) như Toán học (đáp án đúng/sai) hoặc Code (chạy test case pass/fail).
    *   Cho mô hình tự sinh ra chuỗi suy luận.
    *   Nếu đáp án cuối cùng đúng -> Thưởng (Reward).
    *   Mô hình tự học cách suy luận để đạt được phần thưởng mà không cần con người dạy từng bước.

---

## 5. GRPO (Group Relative Policy Optimization)
Đây là thuật toán RL chủ đạo để huấn luyện DeepSeek-R1, cải tiến từ PPO (Proximal Policy Optimization).

**Khác biệt chính với PPO:**
*   **PPO:** Cần một mô hình *Value Function (Critic)* to đùng (bằng kích thước Policy model) để ước lượng lợi thế (Advantage). Rất tốn VRAM và chậm.
*   **GRPO:** Loại bỏ Value Function (Critic).
    *   Thay vào đó, sinh ra một nhóm (Group) các câu trả lời cho cùng một câu hỏi.
    *   Tính lợi thế (Advantage) của mỗi câu trả lời bằng cách so sánh nó với điểm trung bình của cả nhóm.
    *   *Ưu điểm:* Tiết kiệm bộ nhớ, huấn luyện nhanh hơn, ổn định hơn.

---

## 6. DeepSeek-R1 Pipeline
Quy trình tạo ra DeepSeek-R1 (mô hình Reasoning mã nguồn mở mạnh nhất hiện nay):

1.  **Cold Start (Khởi động lạnh):** SFT trên một lượng nhỏ dữ liệu CoT chất lượng cao để mô hình biết định dạng `<think>`.
2.  **Reasoning RL (R1-Zero):** Chạy RL (GRPO) trên quy mô lớn với các bài toán Toán/Code. Mô hình tự phát triển khả năng suy luận vượt trội (Aha moment), nhưng ngôn ngữ có thể bị lộn xộn.
3.  **Rejection Sampling & SFT:** Dùng checkpoint tốt nhất từ bước 2 để sinh ra dữ liệu suy luận sạch đẹp, lọc bỏ các mẫu sai/xấu. Dùng dữ liệu này để SFT lại mô hình Base (R1).
4.  **All-scenario RL:** Chạy RL vòng cuối cùng để căn chỉnh (align) mô hình cho cả các tác vụ không phải suy luận (viết lách, tóm tắt) và đảm bảo an toàn (Safety).

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠](aero_llm_00_overview.md) | [Xem bài viết →](aero_llm_00_overview.md) |
| [Lecture 1: Transformer Architecture 🤖](aero_llm_01_transformer.md) | [Xem bài viết →](aero_llm_01_transformer.md) |
| [Lecture 2: Transformer Tricks & BERT 🛠️](aero_llm_02_transformer_tricks.md) | [Xem bài viết →](aero_llm_02_transformer_tricks.md) |
| [Lecture 3: Large Language Models (LLMs) & Inference 🚀](aero_llm_03_large_language_models.md) | [Xem bài viết →](aero_llm_03_large_language_models.md) |
| [Lecture 4: LLM Training - Pre-training 🏋️](aero_llm_04_training_pretraining.md) | [Xem bài viết →](aero_llm_04_training_pretraining.md) |
| [Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️](aero_llm_05_tuning_peft.md) | [Xem bài viết →](aero_llm_05_tuning_peft.md) |
| 📌 **[Lecture 6: LLM Reasoning 🧠](aero_llm_06_reasoning.md)** | [Xem bài viết →](aero_llm_06_reasoning.md) |
| [Lecture 7: Agentic LLMs & Tool Use 🛠️](aero_llm_07_agentic_llms.md) | [Xem bài viết →](aero_llm_07_agentic_llms.md) |
| [Lecture 8: LLM Evaluation ⚖️](aero_llm_08_evaluation.md) | [Xem bài viết →](aero_llm_08_evaluation.md) |
| [Lecture 9: Recap & Current Trends 🔮](aero_llm_09_trends.md) | [Xem bài viết →](aero_llm_09_trends.md) |
| [🛠️ Top 12 Repo Quan Trọng Cho AI Engineer Tối Ưu LLM](aero_llm_10_essential_tools.md) | [Xem bài viết →](aero_llm_10_essential_tools.md) |
| [Chương 1: Tổng Quan Về Large Language Models (LLMs) 🧠](aero_llm_chapter01_overview_detailed.md) | [Xem bài viết →](aero_llm_chapter01_overview_detailed.md) |
| [Chương 2: 5 Trụ Cột Của Việc Huấn Luyện LLMs 🏛️](aero_llm_chapter02_5pillars_part1.md) | [Xem bài viết →](aero_llm_chapter02_5pillars_part1.md) |
| [Chương 2: 5 Trụ Cột - Part 2 (Evaluation & Systems)](aero_llm_chapter02_5pillars_part2.md) | [Xem bài viết →](aero_llm_chapter02_5pillars_part2.md) |
| [Chương 3: Pre-training → Post-training Pipeline 🔄](aero_llm_chapter03_training_pipeline.md) | [Xem bài viết →](aero_llm_chapter03_training_pipeline.md) |
| [Chương 4 & 5: Mechanisms & Evaluation 🔧📊](aero_llm_chapter04_05_mechanisms_eval.md) | [Xem bài viết →](aero_llm_chapter04_05_mechanisms_eval.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
