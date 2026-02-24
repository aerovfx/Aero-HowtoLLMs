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
