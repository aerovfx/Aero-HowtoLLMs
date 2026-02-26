
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
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
