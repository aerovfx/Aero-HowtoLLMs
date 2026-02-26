
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
# CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠

> **Tổng hợp và biên soạn từ bài giảng CS229 - Machine Learning (Stanford).**
> Tài liệu này tóm tắt các nguyên lý cốt lõi, kiến trúc và quy trình huấn luyện các mô hình ngôn ngữ lớn (Large Language Models) hiện đại.

---

## 📚 Mục Lục

1. [Chương 1: Tổng Quan Về LLMs](#chương-1-tổng-quan-về-llms) | [📖 Chi tiết](aero_LLM_chapter01_overview_detailed.md)
2. [Chương 2: 5 Trụ Cột](#chương-2-5-trụ-cột-của-việc-huấn-luyện) | [📖 Part 1](aero_LLM_chapter02_5pillars_part1.md) | [📖 Part 2](aero_LLM_chapter02_5pillars_part2.md)
3. [Chương 3: Pre-training → Post-training](#chương-3-quy-trình-từ-pre-training-đến-post-training) | [📖 Chi tiết](aero_LLM_chapter03_training_pipeline.md)
4. [Chương 4 & 5: Mechanisms & Evaluation](#chương-4-cơ-chế-hoạt-động-autoregressive--tokenization) | [📖 Chi tiết](aero_LLM_chapter04_05_mechanisms_eval.md)

---

## 🎯 GPT-4 Interactive Visualization

**Xem trực tiếp kiến trúc GPT-4:**

```bash
cd llm_viz && npm run dev
# → http://localhost:3002/llm

✅ **100% Vietnamese** | ✅ **MoE Expert Grid** | ✅ **Interactive**

---

## Chương 1: Tổng Quan Về LLMs

**LLM** = Mô hình phân phối xác suất trên chuỗi tokens, dựa trên **Transformer**.

**Ví dụ:** GPT-4 (1.76T params), Claude 3 Opus, Gemini Ultra, Llama 3

[📖 **Chi tiết**](aero_LLM_chapter01_overview_detailed.md)

---

## Chương 2: 5 Trụ Cột Của Việc Huấn Luyện

1. **Architecture:** MoE Transformer (GPT-4)
2. **Loss:** Cross-Entropy + RLHF (PPO)
3. **Data:** 13T tokens (web, books, code)
4. **Evaluation:** MMLU 86%, HumanEval 67%
5. **Systems:** 10K+ H100 GPUs, $100M cost

> 💡 Industry focus: **Data (35%) + Systems (10%) + Evaluation (15%)**

[📖 **Part 1**](aero_LLM_chapter02_5pillars_part1.md) | [📖 **Part 2**](aero_LLM_chapter02_5pillars_part2.md)

---

## Chương 3: Pre-training → Post-training

**Pipeline:**
Pre-training (100 days, $100M)
  ↓
SFT - 50K examples (3 days, $10K)
  ↓
RLHF - Human preferences (1 week, $50K)
  ↓
ChatGPT ✅

[📖 **Chi tiết**](aero_LLM_chapter03_training_pipeline.md)

---

## Chương 4: Autoregressive & Tokenization

**Autoregressive:** $P(x)$ = ∏ P(xᵢ | x₁...xᵢ₋₁)  
**Tokenization:** BPE, ~100K vocab  
**Issues:** Numbers, indentation, non-English

---

## Chương 5: Evaluation

1. **Perplexity:** ~8 (GPT-4)
2. **Benchmarks:** MMLU, HumanEval, GSM8K
3. **Human Eval:** Helpful, Honest, Harmless
4. **Production:** < 500ms latency

[📖 **Chi tiết**](aero_LLM_chapter04_05_mechanisms_eval.md)

---

*Biên soạn bởi Pixibot - Stanford CS229*  
*GPT-4 Visualization: ✅ Complete*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠](aero_llm_00_overview.md)** | [Xem bài viết →](aero_llm_00_overview.md) |
| [Lecture 1: Transformer Architecture 🤖](aero_llm_01_transformer.md) | [Xem bài viết →](aero_llm_01_transformer.md) |
| [Lecture 2: Transformer Tricks & BERT 🛠️](aero_llm_02_transformer_tricks.md) | [Xem bài viết →](aero_llm_02_transformer_tricks.md) |
| [Lecture 3: Large Language Models (LLMs) & Inference 🚀](aero_llm_03_large_language_models.md) | [Xem bài viết →](aero_llm_03_large_language_models.md) |
| [Lecture 4: LLM Training - Pre-training 🏋️](aero_llm_04_training_pretraining.md) | [Xem bài viết →](aero_llm_04_training_pretraining.md) |
| [Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️](aero_llm_05_tuning_peft.md) | [Xem bài viết →](aero_llm_05_tuning_peft.md) |
| [Lecture 6: LLM Reasoning 🧠](aero_llm_06_reasoning.md) | [Xem bài viết →](aero_llm_06_reasoning.md) |
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
