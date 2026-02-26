
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
# Chương 2: 5 Trụ Cột - Part 2 (Evaluation & Systems)

> **Tiếp theo từ Part 1**

---

## Trụ Cột 4: Evaluation (Đánh Giá)

### **"You Can't Improve What You Don't Measure"**

### **Tại Sao Evaluation Quan Trọng?**

1. **Progress tracking:** Biết model có đang học không
2. **Model comparison:** A tốt hơn B ở đâu?
3. **Debug:** Tìm weakness để cải thiện
4. **Business value:** ROI, user satisfaction

### **Levels of Evaluation**

Level 1: Perplexity (Training metric)
  ↓
Level 2: Academic Benchmarks (MMLU, HumanEval)
  ↓
Level 3: Human Evaluation (Quality, safety)
  ↓
Level 4: Real-world Usage (Production metrics)

---

### **A. Perplexity**

**Định nghĩa:**

PPL = exp(-1/N ∑ᵢ log P(xᵢ  \mid  x₁,...,xᵢ₋₁))

**Ý nghĩa:**
- Độ "bối rối" của model khi dự đoán
- **Thấp hơn = Tốt hơn**
- PPL = 10 → Model "surprise" ít hơn PPL = 100

**Example:**
```python
# Sentence: "The cat sat on the mat"

$$
probs = [0.8, 0.6, 0.9, 0.7, 0.5, 0.8]  # Probabilities ppl = exp(-mean([log(p) for p in probs])) # ppl ≈ 1.8 (very good) **Historical Trends:** \mid Year \mid Model \mid Perplexity (WikiText-103) \mid |------|-------|---------------------------| \mid 2017 \mid LSTM \mid ~70 \mid \mid 2018 \mid GPT-1 \mid ~37 \mid \mid 2019 \mid GPT-2 \mid ~18 \mid \mid 2020 \mid GPT-3 \mid ~15 \mid \mid 2023 \mid GPT-4 \mid ~8 (estimated) \mid **Limitations:** - ❌ Doesn't measure reasoning - ❌ Doesn't capture safety - ❌ Can be gamed (memorization) --- ### **B. Academic Benchmarks** #### **1. MMLU (Massive Multitask Language Understanding)** **What:** 57 subjects, multiple-choice questions **Subjects:** - STEM: Math, Physics, Chemistry, CS - Humanities: History, Philosophy, Law - Social Sciences: Psychology, Economics - Other: Medicine, Business **Format:** Question: What is the primary function of ribosomes? A) DNA replication B) Protein synthesis C) Cell division D) Energy production Answer: B **GPT-4 Performance:** - GPT-3.5: 70.0% - **GPT-4: 86.4%** (human expert ~90%) **Leaderboard (2024):** 1. GPT-4: 86.4% 2. Claude 3 Opus: 86.8% 3. Gemini Ultra: 90.0% #### **2. HumanEval (Code Generation)** **What:** 164 Python programming problems **Format:** ```python def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. """ # Model generates code here **Metrics:** - Pass@1: % correct on first try - Pass@10: % correct in 10 tries **Results:** \mid Model \mid Pass@1 \mid |-------|--------| \mid GPT-3 \mid 0% \mid \mid Codex \mid 28.8% \mid \mid GPT-3.5 \mid 48.1% \mid \mid **GPT-4** \mid **67.0%** \mid \mid Claude 3 Opus \mid 84.9% \mid #### **3. Other Important Benchmarks** \mid Benchmark \mid Focus \mid Examples \mid |-----------|-------|----------| \mid **GSM8K** \mid Math reasoning \mid 8K grade school problems \mid \mid **HellaSwag** \mid Commonsense \mid Sentence completion \mid \mid **TruthfulQA** \mid Truthfulness \mid Avoiding misconceptions \mid \mid **BBHard** \mid Hard reasoning \mid 23 challenging tasks \mid \mid **DROP** \mid Reading comp \mid Complex reasoning over text \mid --- ### **C. Human Evaluation** **Why needed:** - Benchmarks can be memorized - Real-world tasks are open-ended - Subjective quality matters **Evaluation Criteria:** 1. **Helpfulness** - Did it answer the question? - Is the answer complete? - Is it actionable? 2. **Harmlessness** - No toxic content - No illegal advice - No personal attacks 3. **Honesty** - Admits uncertainty - Doesn't hallucinate - Cites sources when applicable **Evaluation Process:** 1. Sample Generation ├── User prompts (diverse topics) ├── Generate responses └── Multiple models (A/B/C) 2. Human Rating ├── Likert scale (1-5) ├── Pairwise comparison └── Preference ranking 3. Analysis ├── Inter-rater agreement ├── Confidence intervals └── Statistical significance 4. Iterate └── Fix common failures **Example (ChatGPT Eval):** Prompt: "Explain quantum computing to a 5-year-old" GPT-3.5: [Technical jargon, not age-appropriate] Rating: 2/5 GPT-4: "Imagine a magic computer that can be in many places at once, like being in your room AND the kitchen at the same time..." Rating: 5/5 --- ### **D. Real-World Metrics** **Production KPIs:** \mid Metric \mid Definition \mid Target \mid |--------|------------|--------| \mid **Latency** \mid Time to first token \mid < 500ms \mid \mid **Throughput** \mid Tokens/sec \mid > 100 \mid \mid **Cost** \mid /1K tokens \mid  < 0.01 \mid \mid **Uptime** \mid Availability % \mid > 99.9% \mid \mid **User Satisfaction** \mid Thumbs up % \mid > 80% \mid **Business Metrics:** - Retention rate - Engagement msgs/user/day - Revenue (subscriptions, API usage) --- ## Trụ Cột 5: Systems (Hệ Thống) ### **"Training is a Systems Problem"** **Reality:** - GPT-3: 355 GPU-years - GPT-4: ~2000 GPU-years - Gemini Ultra: Estimated 10,000+ GPU-years **Cost:** - GPT-3: ~4M - GPT-4: ~100M - Training run can cost more than entire startups! --- ### **A. Hardware** #### **GPUs for Training** \mid GPU \mid Memory \mid FP16 TFLOPS \mid Price \mid Used By \mid |-----|--------|-------------|-------|---------| \mid **A100** \mid 80GB \mid 312 \mid ~15K \mid GPT-3, most labs \mid \mid **H100** \mid 80GB \mid 1000 \mid ~40K \mid **GPT-4, Gemini** \mid \mid **MI250X** (AMD) \mid 128GB \mid 383 \mid ~12K \mid Stable Diffusion XL \mid \mid **TPU v4** \mid 32GB HBM |Variable \mid Google only \mid PaLM, Gemini \mid **GPT-4 Cluster (estimated):** 10,000× H100 GPUs
$$

├── 8× GPUs per node = 1,250 nodes

$$
├── NVLink: 600 GB/s inter-GPU ├── InfiniBand: 400 Gb/s networking └── Total compute: ~100,000 petaFLOPS **Cost per hour:**
$$

10,000 H100 × 3/hr = 30,000/hour

100 days training = 72 million (compute only!)

$$
#### **Memory Hierarchy** L1 Cache (KB)      ← 1000× faster, tiny ↓ L2 Cache (MB)      ← 100× faster, small ↓ GPU RAM (80GB)     ← 10× faster, limited ↓ CPU RAM (1TB)      ← Baseline ↓ SSD (10TB)         ← 10× slower ↓ Network Sto18_rage    ← 100× slower **Challenge:** Model doesn't fit in GPU RAM! --- ### **B. Parallelization Strategies** #### **1. Data Parallelism** GPU 0: Batch 0 → Forward → Backward → Grad₀ GPU 1: Batch 1 → Forward → Backward → Grad₁ GPU 2: Batch 2 → Forward → Backward → Grad₂ ↓ All-Reduce (Average gradients) ↓ Update weights (synchronized) **Pros:** Simple, linear scaling **Cons:** Requires full model on each GPU #### **2. Model Parallelism (Tensor Parallelism)** Layer splits across GPUs: GPU 0: [A] ──→ [B] ↓ GPU 1: [C] ──→ [D] All-to-All communication **Example (GPT-4):** ```python # Split attention across 8 GPUs Q = split(Q, dim=heads, n_splits=8)  # Each GPU gets 16/8 = 2 heads
$$

K = split(K, dim=heads, n_splits=8)

$$
V = split(V, dim=heads, n_splits=8) **Pros:** Handles huge models **Cons:** High communication overhead #### **3. Pipeline Parallelism** GPU 0: Layer 0-29   → Forward Batch 0 → Forward Batch 1 → ↓ GPU 1: Layer 30-59  → (wait)          → Forward Batch 0 → ↓ GPU 2: Layer 60-89  → (wait)          → (wait)          → ↓ GPU 3: Layer 90-119 → (idle)          → (idle)          → **GPipe / 1F1B:** - Micro-batches to reduce bubbles - Backward pass interleaved **Pros:** Good for very deep models **Cons:** Bubble time (idle GPUs) #### **4. 3D Parallelism (ZeRO)** **Combines all three:** ZeRO Stage 1: Partition optimizer states ZeRO Stage 2: + Partition gradients ZeRO Stage 3: + Partition model weights **Memory Savings:**
$$

Before: 1.76T params × 16 bytes = 28 TB (per GPU!)

After ZeRO-3: 28 TB / 10,000 GPUs = 2.8 GB per GPU ✅

$$
**Used by:** - GPT-4 (DeepSpeed ZeRO) - Megatron-LM (NVIDIA) - FSDP (PyTorch) --- ### **C. Training Infrastructure** **Full Stack:** Application Layer ├── PyTorch / JAX ├── DeepSpeed / Megatron └── Model code Training Framework ├── Distributed training ├── Mixed precision ├── Gradient accumulation └── Checkpointing Systems Layer ├── NCCL (GPU communication) ├── InfiniBand (networking) └── Sto18_rage (NVMe, Lustre) Hardware ├── 10,000+ H100 GPUs ├── High-speed interconnects └── Cooling & power **GPT-4 Training Pipeline:** ```python # Pseudo-code model = GPT4(params=1.76T)
$$

optimizer = AdamW(lr=6e-4)

$$
scaler = GradScaler()  # Mixed precision # 3D Parallelism model = apply_tensor_parallel(model, tp_size=8)
$$

model = apply_pipeline_parallel(model, pp_size=16)

$$
model = apply_data_parallel(model, dp_size=78) # Total: 8 × 16 × 78 ≈ 10,000 GPUs for epoch in range(3):  # 3 epochs × 13T tokens for batch in dataloader: with autocast(dtype=bfloat16):
$$

output = model(batch)

$$
loss = cross_entropy(output, targets) scaler.scale(loss).backward() scaler.step(optimizer) scaler.update() if step % checkpoint_interval == 0: save_checkpoint(model, optimizer, step) --- ### **D. Optimization Techniques** **1. Gradient Checkpointing:** ```python # Trade compute for memory # Recompute activations during backward model = checkpoint_sequential(model, segments=4) # 4× less memory, 20% slower **2. Flash Attention:** ```python # Fused attention kernel # 2-4× faster, less memory from flash_attn import flash_attn_func attn_output = flash_attn_func(Q, K, V) **3. Quantization:** ```python # Train in INT8 model = quantize_dynamic(model, dtype=torch.qint8) # 2× faster, 4× less memory --- ## Academia vs Industry ### **Focus Distribution** \mid Pillar \mid Academia \mid Industry \mid |--------|----------|----------| \mid Architecture \mid **80%** \mid 20% \mid \mid Loss/Algorithm \mid 15% \mid 20% \mid \mid Data \mid 3% \mid **35%** \mid \mid Evaluation \mid 2% \mid **15%** \mid \mid Systems \mid 0% \mid **10%** \mid **Why the difference?** **Academia:** - New architectures → Papers - Limited compute budget - Public datasets - Leaderboard chasing **Industry:** - Product quality → Revenue - Massive compute access - Proprietary data advantage - Real user feedback --- ## 🎯 Key Takeaways 1. ✅ **5 Pillars ALL matter** - ignoring any one = failure
$$

