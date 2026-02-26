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
