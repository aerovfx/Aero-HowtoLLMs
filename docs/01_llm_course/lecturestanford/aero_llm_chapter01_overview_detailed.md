
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
# Chương 1: Tổng Quan Về Large Language Models (LLMs) 🧠

> **Khóa học:** CS229 - Machine Learning (Stanford)  
> **Chương:** 1/5 - Foundation & Overview  
> **Mục tiêu:** Hiểu khái niệm cơ bản về LLMs và vị trí của chúng trong AI

---

## 📚 Nội Dung Chương

1. [Định Nghĩa LLM](#định-nghĩa-llm)
2. [Kiến Trúc Transformer](#kiến-trúc-transformer)
3. [Các Mô Hình Tiêu Biểu](#các-mô-hình-tiêu-biểu)
4. [GPT-4: Case Study](#gpt-4-case-study)
5. [Tại Sao LLMs Quan Trọng](#tại-sao-llms-quan-trọng)

---

## 1. Định Nghĩa LLM

### **LLM là gì?**

**Large Language Model** (Mô hình Ngôn ngữ Lớn) là các mô hình phân phối xác suất trên các chuỗi từ (sequences of tokens).

**Định nghĩa toán học:**

P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)

Nói cách khác:
- **Input:** Chuỗi văn bản (text sequence)
- **Output:** Xác suất của từ tiếp theo (next token probability)
- **Mục tiêu:** Mô hình hóa ngôn ngữ tự nhiên theo cách máy tính có thể "hiểu"

### **"Large" có nghĩa là gì?**

| Thế hệ | Số tham số | Ví dụ |
|--------|------------|-------|
| **Small** | < 1B | GPT-2 (1.5B) |
| **Medium** | 1B - 10B | Llama 2 (7B) |
| **Large** | 10B - 100B | GPT-3 (175B) |
| **Extra Large** | > 100B | GPT-4 (1.76T) |

> 💡 **Lưu ý:** "Large" không chỉ về số lượng tham số mà còn về:
> - Dữ liệu training (trillions of tokens)
> - Khả năng emergent (xuất hiện tự nhiên)
> - Context window (độ dài văn bản xử lý được)

---

## 2. Kiến Trúc Transformer

### **Tại sao Transformer?**

Trước Transformer (2017), các mô hình ngôn ngữ sử dụng:
- **RNN** (Recurrent Neural Networks): Chậm, khó train
- **LSTM** (Long Short-Term Memory): Tốt hơn nhưng vẫn tuần tự
- **CNN**: Không phù hợp với sequences dài

**Transformer** giải quyết vấn đề bằng **Self-Attention**:
Attention(Q, K, V) = softmax(QKᵀ/√d) × V

### **Cấu Trúc Transformer Block**

Input Embedding
    ↓
[Position Embedding] ─┐
    ↓                 │
Layer Norm            │
    ↓                 │
Multi-Head Attention  │
    ↓                 │
Add & Norm ←──────────┘ (Residual Connection)
    ↓
Layer Norm ───────────┐
    ↓                 │
Feed-Forward (MLP)    │
    ↓                 │
Add & Norm ←──────────┘ (Residual Connection)
    ↓

$$
Repeat N times
$$

↓
Output Layer

### **Thành phần chính:**

1. **Embeddings:**
   - Token Embedding: Chuyển từ → vector
   - Position Embedding: Thêm thông tin vị trí

2. **Self-Attention:**
   - Q (Query): "Tôi đang tìm gì?"
   - K (Key): "Tôi cung cấp thông tin gì?"
   - V (Value): "Giá trị thực tế tôi mang"

3. **Feed-Forward (MLP):**
   - Thường expand 4× (C → 4C → C)
   - Activation: GELU (GPT), ReLU (BERT)

4. **Layer Normalization:**
   - Ổn định training
   - Normalize theo từng column

5. **Residual Connections:**
   - Giúp gradient flow
   - Cho phép train mô hình sâu

---

## 3. Các Mô Hình Tiêu Biểu

### **A. GPT Series (OpenAI)**

| Model | Year | Params | Context | Key Features |
|-------|------|--------|---------|--------------|
| GPT-1 | 2018 | 117M | 512 | Proof of concept |
| GPT-2 | 2019 | 1.5B | 1024 | Zero-shot learning |
| GPT-3 | 2020 | 175B | 2048 | Few-shot learning |
| GPT-3.5 | 2022 | ~175B | 4096 | ChatGPT base |
| **GPT-4** | 2023 | **1.76T** | **32K-128K** | **MoE, Multimodal** |

**Đặc điểm:**
- **Architecture:** Decoder-only Transformer
- **Training:** Autoregressive (predict next token)
- **Strength:** Text generation, reasoning

### **B. Claude (Anthropic)**

| Model | Year | Context | Key Features |
|-------|------|---------|--------------|
| Claude 1 | 2022 | 9K | RLHF focused |
| Claude 2 | 2023 | 100K | Long context |
| **Claude 3** | 2024 | **200K** | **Opus/Sonnet/Haiku** |

**Đặc điểm:**
- **Focus:** Safety, helpfulness, harmlessness (HHH)
- **Strength:** Long documents, technical writing

### **C. Gemini (Google)**

| Model | Year | Params | Key Features |
|-------|------|--------|--------------|
| Gemini Pro | 2023 | ~175B | Production |
| **Gemini Ultra** | 2024 | **~1.5T** | **SOTA multimodal** |

**Đặc điểm:**
- **Multimodal:** Text, image, video, audio
- **Integration:** Google ecosystem

### **D. Llama (Meta)**

| Model | Year | Open Source | Key Features |
|-------|------|-------------|--------------|
| Llama | 2023 | ✅ | Research only |
| **Llama 2** | 2023 | ✅ | **Commercial use** |
| **Llama 3** | 2024 | ✅ | **400B, multilingual** |

**Đặc điểm:**
- **Open weights:** Available for download
- **Community:** Huge ecosystem (Alpaca, Vicuna, etc.)

---

## 4. GPT-4: Case Study

### **Kiến Trúc GPT-4**

**🔥 Mixture of Experts (MoE):**

Input
  ↓
Embedding
  ↓
┌─────────────────────┐
│ Transformer Block   │
│   ↓                 │
│ Self-Attention      │
│   ↓                 │
│ [Router] ──→ Top-K  │  ← Chọn 2/8 experts
│   ↓                 │
│ ┌───┬───┬───┬───┐   │
│ │E0 │E1 │E2 │E3 │   │  ← Expert grid 2×4
│ │E4 │E5 │E6 │E7 │   │
│ └───┴───┴───┴───┘   │
│   ↓                 │
│ Aggregation         │
└─────────────────────┘
  ↓

$$
Repeat 120 layers
$$

↓
Output

**Thông số kỹ thuật:**

| Metric | Value |
|--------|-------|
| **Total Parameters** | ~1.76 Trillion |
| **Active Parameters** | ~220B per forward pass |
| **Experts per layer** | 8 (MoE) |
| **Active experts** | 2 (Top-K) |
| **Layers** | ~120 |
| **Embedding dimension** | ~18,432 |
| **Attention heads** | ~128 |
| **Context window** | 32K (standard), 128K (extended) |
| **Training tokens** | ~13 Trillion |

### **So sánh GPT-3 vs GPT-4:**

| Feature | GPT-3 | GPT-4 |
|---------|-------|-------|
| Architecture | Dense | **MoE (Sparse)** |
| Total params | 175B | **1.76T** |
| Active params | 175B | **~220B** |
| Multimodal | ❌ | **✅ (Vision)** |
| Context | 2K-4K | **32K-128K** |
| Training cost | ~$4M | **~$100M** |

### **Ưu điểm MoE:**

1. **Hiệu quả hơn:**
   - Only 2/8 experts active → Save compute
   - 1.76T total, ~220B active → 8× cheaper than dense

2. **Chuyên môn hóa:**
   - Expert 0: Math, logic
   - Expert 1: Creative writing
   - Expert 2: Code generation
   - ...

3. **Scalability:**
   - Dễ mở rộng (add more experts)
   - Parallel training

---

## 5. Tại Sao LLMs Quan Trọng?

### **A. Emergent Abilities (Khả năng Xuất Hiện)**

Khi model đủ lớn (>100B params), xuất hiện các khả năng mới:

1. **Few-shot Learning:** Học từ vài ví dụ
2. **Chain-of-Thought:** Suy luận từng bước
3. **Reasoning:** Giải toán, logic
4. **Code Generation:** Viết code chất lượng cao

### **B. Ứng Dụng Thực Tế**

| Domain | Use Cases |
|--------|-----------|
| **Programming** | GitHub Copilot, Code review |
| **Writing** | Content generation, editing |
| **Education** | Tutoring, explanations |
| **Research** | Literature review, summarization |
| **Business** | Customer service, automation |
| **Creative** | Storytelling, brainstorming |

### **C. Tác Động Kinh Tế**

- **Productivity:** McKinsey: +$4.4T/year by 2030
- **Jobs:** 300M jobs affected (transformed, not replaced)
- **Industry:** Every sector integrating AI

### **D. Research Direction**

**Hot topics:**

1. **Efficiency:**
   - Quantization (INT8, INT4)
   - Pruning
   - Distillation

2. **Multimodality:**
   - Text + Vision + Audio
   - Unified embeddings

3. **Reasoning:**
   - Chain-of-Thought
   - Tree-of-Thought
   - Self-consistency

4. **Safety:**
   - Alignment
   - RLHF
   - Constitutional AI

---

## 📊 Visualization Link

**Xem trực quan GPT-4 Architecture:**

```bash
cd llm_viz
npm run dev
# Open: http://localhost:3002/llm
# Select: GPT-4 model

**Các phần có thể explore:**
- ✅ Token & Position Embeddings
- ✅ Layer Normalization
- ✅ Self-Attention Mechanism
- ✅ MoE Expert Grid (2×4)
- ✅ Router & Top-K Selection
- ✅ MLP (Feed-Forward)
- ✅ Residual Connections
- ✅ Output Layer & Logits

**Ngôn ngữ:** 🇻🇳 Vietnamese (100% localized)

---

## 🎯 Key Takeaways

1. ✅ **LLM = Probability model** over token sequences
2. ✅ **Transformer architecture** is the foundation
3. ✅ **GPT-4 uses MoE** for efficiency at scale
4. ✅ **Emergent abilities** appear at 100B+ params
5. ✅ **Multimodality** is the future direction

---

## 📚 Đọc Thêm

**Papers:**
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Original Transformer
- [GPT-3 (2020)](https://arxiv.org/abs/2005.14165) - Language Models are Few-Shot Learners
- [GPT-4 Technical Report (2023)](https://arxiv.org/abs/2303.08774)

**Courses:**
- Stanford CS229: Machine Learning
- Stanford CS224N: NLP with Deep Learning
- Fast.ai: Practical Deep Learning

**Interactive:**
- Our visualization tool $llm_viz$
- Transformer Explainer (Poloclub)
- LLM Visualization (bbycroft)

---

**Next:** [Chương 2: 5 Trụ Cột Của Việc Huấn Luyện →](./aero_LLM_chapter02_5pillars_part1.md)

---

*Biên soạn bởi Pixibot - Based on Stanford CS229*  
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
| 📌 **[Chương 1: Tổng Quan Về Large Language Models (LLMs) 🧠](aero_llm_chapter01_overview_detailed.md)** | [Xem bài viết →](aero_llm_chapter01_overview_detailed.md) |
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
