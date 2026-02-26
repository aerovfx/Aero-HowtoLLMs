
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
```

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
```
Pre-training (100 days, $100M)
  ↓
SFT - 50K examples (3 days, $10K)
  ↓
RLHF - Human preferences (1 week, $50K)
  ↓
ChatGPT ✅
```

[📖 **Chi tiết**](aero_LLM_chapter03_training_pipeline.md)

---

## Chương 4: Autoregressive & Tokenization

**Autoregressive:** P(x) = ∏ P(xᵢ | x₁...xᵢ₋₁)  
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
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
