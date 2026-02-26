
<!-- Aero-Navigation-Start -->
[🏠 Home](../../../index.md) > [01 llm course](../../index.md) > [lecturestanford](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../../index.md)
- [📚 Module 01: LLM Course](../../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Chương 4 & 5: Mechanisms & Evaluation 🔧📊

> **CS229 Stanford** | Chương 4-5/5  
> **Autoregressive, Tokenization & Evaluation**

---

## CHƯƠNG 4: Cơ Chế Hoạt Động

### **1. Autoregressive Generation**

**Formula:**
```
P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)
```

**Example:**
```
Input: "The cat"
Step 1: P("sat" | "The cat") = 0.8 → Output: "sat"
Step 2: P("on" | "The cat sat") = 0.9 → Output: "on"
Step 3: P("the" | "The cat sat on") = 0.95 → Output: "the"
...

Result: "The cat sat on the mat"
```

**Hạn chế:**
- Slow (sequential, not parallel)
- Can't "look ahead"
- Expensive at scale

---

### **2. Tokenization**

**Why?** Computers understand numbers, not words.

**BPE Algorithm:**
```
1. Start: ['h', 'e', 'l', 'l', 'o'] 
2. Merge frequent: 'l'+'l' → 'll'
3. Result: ['h', 'e', 'll', 'o']
4. Repeat...
```

**GPT-4 Tokenizer:**
- Vocab size: ~100K tokens
- Ave18-RAGe: 1 token ≈ 0.75 words
- Handles 100+ languages

**Common Issues:**
```python
# Numbers get split weirdly
"327" → ["3", "27"] ❌  # Bad for math

# Indentation problems (old models)
"    def foo():" → ["  ", "  ", "def", " foo", "():"]

# Non-English struggles
"你好" (Chinese) → Multiple f18-RAGments
```

---

## CHƯƠNG 5: Đánh Giá

### **1. Perplexity**

```
PPL = exp(-1/N ∑ log P(xᵢ | context))
```

**Lower = Better**

| Model | PPL (WikiText) |
|-------|----------------|
| LSTM (2017) | ~70 |
| GPT-2 (2019) | ~18 |
| **GPT-4 (2023)** | **~8** |

---

### **2. Benchmarks**

#### **MMLU (Knowledge)**
- 57 subjects
- Multiple choice
- GPT-4: **86.4%** (human expert ~90%)

#### **HumanEval (Coding)**
- 164 Python problems
- GPT-4: **67.0% pass@1**

#### **GSM8K (Math)**
- Grade school math
- GPT-4: **92.0%**

---

### **3. Human Evaluation**

**Criteria:**
1. **Helpful:** Did it answer well?
2. **Honest:** No hallucinations?
3. **Harmless:** No toxic content?

**Process:**
```
Generate responses → Humans rate → Statistical analysis
```

---

### **4. Production Metrics**

| Metric | Target |
|--------|--------|
| Latency | < 500ms |
| Throughput | > 100 tok/s |
| Cost | < $0.01/1K tokens |
| Uptime | > 99.9% |

---

## 🎯 Summary: Full LLM Stack

```
┌─────────────────────────────┐
│  User Prompt                │
└──────────┬──────────────────┘
           ↓
     [Tokenization]
           ↓
┌─────────────────────────────┐
│  Embedding Layer            │
│  Position + Token           │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│  Transformer Blocks × 120   │
│  ├─ Layer Norm              │
│  ├─ Self-Attention          │
│  ├─ MoE (8 experts, Top-2)  │
│  └─ Residual                │
└──────────┬──────────────────┘
           ↓
     [Final LN + Linear]
           ↓
     [Softmax → Probs]
           ↓
     [Sample Next Token]
           ↓
     [Autoregressive Loop]
           ↓
┌─────────────────────────────┐
│  Generated Response         │
└─────────────────────────────┘
```

---

## ✅ Key Takeaways (All 5 Chapters)

**Ch 1: Overview**
- LLM = Probability model on token sequences
- Transformer architecture dominates
- GPT-4 = MoE (1.76T params)

**Ch 2: 5 Pillars**
- Architecture, Loss, Data, Evaluation, Systems
- Industry focuses on Data + Systems
- $100M training requires 10K+ GPUs

**Ch 3: Training**
- Pre-training (13T tokens, 100 days)
- SFT (50K examples, 3 days)
- RLHF (preferences, 1 week)

**Ch 4: Mechanisms**
- Autoregressive = sequential generation
- Tokenization = text ↔ numbers
- BPE handles multiple languages

**Ch 5: Evaluation**
- Perplexity: ~8 (SOTA)
- Benchmarks: MMLU 86%, HumanEval 67%
- Human eval: Helpful, Honest, Harmless
- Production: < 500ms latency

---

## 🎓 Next Steps

### **Hands-on:**
1. **Explore GPT-4 Visualization:**
   ```bash
   cd llm_viz && npm run dev
   # → http://localhost:3002/llm
   ```

2. **Try Vietnamese Walkthroughs:**
   - Embedding
   - Layer Normalization
   - Self-Attention
   - MoE Routing
   - MLP
   - Output Layer

3. **Experiment:**
   - Adjust temperature
   - Compare models
   - Observe expert selection

### **Further Reading:**
- Original Transformer paper (2017)
- GPT-3 paper (2020)
- GPT-4 Technical Report (2023)
- InstructGPT (RLHF) paper (2022)

### **Build Your Own:**
1. Start small: Train on WikiText
2. Use open models: Llama 2
3. Try fine-tuning: PEFT, LoRA
4. Scale up gradually

---

**🎉 Congratulations!**  
You now understand the complete LLM stack from architecture to deployment!

---

*Biên soạn bởi Pixibot - Stanford CS229*  
*Last updated: 2026-02-15*
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
| [Chương 3: Pre-training → Post-training Pipeline 🔄](aero_llm_chapter03_training_pipeline.md) | [Xem bài viết →](aero_llm_chapter03_training_pipeline.md) |
| 📌 **[Chương 4 & 5: Mechanisms & Evaluation 🔧📊](aero_llm_chapter04_05_mechanisms_eval.md)** | [Xem bài viết →](aero_llm_chapter04_05_mechanisms_eval.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
