
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
# Chương 3: Pre-training → Post-training Pipeline 🔄

> **CS229 Stanford** | Chương 3/5  
> **Từ Base Model → ChatGPT**

---

## 📊 Complete Pipeline

```
13T Tokens Data
      ↓
Pre-training (100 days, $100M)
      ↓
Base Model (knows but doesn't follow)
      ↓
SFT - 50K examples (3 days)
      ↓
Instruction Model
      ↓
RLHF - Reward Model + PPO (1 week)
      ↓
ChatGPT ✅
```

---

## 1. Pre-training

**Input:** 13 trillion tokens  
**Output:** Base model (GPT-4 base)  
**Time:** 100+ days  
**Cost:** ~$100M

**What it learns:**
- Grammar, syntax
- Facts, knowledge
- Code patterns
- Logic, reasoning

**What it can't do:**
- Follow instructions reliably
- Refuse harmful requests
- Format responses nicely

---

## 2. Supervised Fine-Tuning (SFT)

**Input:** ~50K instruction-response pairs  
**Output:** Instruction-following model  
**Time:** 1-3 days  
**Cost:** ~$10K

**Example data:**
```json
{
  "prompt": "Explain photosynthesis simply",
  "response": "Photosynthesis is how plants make food using sunlight..."
}
```

---

## 3. RLHF (Reinforcement Learning)

**Input:** Human preferences  
**Output:** Aligned assistant (ChatGPT)  
**Time:** ~1 week  
**Cost:** ~$50K

**Process:**
1. Collect comparisons (A vs B)
2. Train reward model
3. Optimize policy with PPO

---

## 🎯 Key Takeaways

- ✅ Pre-training: Learn language (expensive!)
- ✅ SFT: Learn to follow instructions
- ✅ RLHF: Align with human values
- ✅ 3 stages = ChatGPT

---

**Next:** [Chương 4: Autoregressive & Tokenization →](./aero_LLM_chapter04_05_mechanisms_eval.md)
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
| [Lecture 6: LLM Reasoning 🧠](aero_llm_06_reasoning.md) | [Xem bài viết →](aero_llm_06_reasoning.md) |
| [Lecture 7: Agentic LLMs & Tool Use 🛠️](aero_llm_07_agentic_llms.md) | [Xem bài viết →](aero_llm_07_agentic_llms.md) |
| [Lecture 8: LLM Evaluation ⚖️](aero_llm_08_evaluation.md) | [Xem bài viết →](aero_llm_08_evaluation.md) |
| [Lecture 9: Recap & Current Trends 🔮](aero_llm_09_trends.md) | [Xem bài viết →](aero_llm_09_trends.md) |
| [🛠️ Top 12 Repo Quan Trọng Cho AI Engineer Tối Ưu LLM](aero_llm_10_essential_tools.md) | [Xem bài viết →](aero_llm_10_essential_tools.md) |
| [Chương 1: Tổng Quan Về Large Language Models (LLMs) 🧠](aero_llm_chapter01_overview_detailed.md) | [Xem bài viết →](aero_llm_chapter01_overview_detailed.md) |
| [Chương 2: 5 Trụ Cột Của Việc Huấn Luyện LLMs 🏛️](aero_llm_chapter02_5pillars_part1.md) | [Xem bài viết →](aero_llm_chapter02_5pillars_part1.md) |
| [Chương 2: 5 Trụ Cột - Part 2 (Evaluation & Systems)](aero_llm_chapter02_5pillars_part2.md) | [Xem bài viết →](aero_llm_chapter02_5pillars_part2.md) |
| 📌 **[Chương 3: Pre-training → Post-training Pipeline 🔄](aero_llm_chapter03_training_pipeline.md)** | [Xem bài viết →](aero_llm_chapter03_training_pipeline.md) |
| [Chương 4 & 5: Mechanisms & Evaluation 🔧📊](aero_llm_chapter04_05_mechanisms_eval.md) | [Xem bài viết →](aero_llm_chapter04_05_mechanisms_eval.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
