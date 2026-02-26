
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
# Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này tập trung vào giai đoạn sau Pre-training: Supervised Fine-Tuning (SFT) để biến mô hình thành trợ lý, và các kỹ thuật Fine-tuning hiệu quả (PEFT/LoRA).

---

## 📚 Mục Lục
1. [Supervised Fine-Tuning (SFT)](#1-supervised-fine-tuning-sft)
2. [Instruction Tuning](#2-instruction-tuning)
3. [Dữ liệu cho SFT](#3-dữ-liệu-cho-sft)
4. [Parameter-Efficient Fine-Tuning (PEFT)](#4-parameter-efficient-fine-tuning-peft)
5. [LoRA (Low-Rank Adaptation)](#5-lora-low-rank-adaptation)
6. [QLoRA (Quantized LoRA)](#6-qlora-quantized-lora)

---

## 1. Supervised Fine-Tuning (SFT)
Sau Pre-training, mô hình giống như một "con vẹt thông thái" - biết rất nhiều nhưng chỉ biết dự đoán từ tiếp theo chứ không biết trả lời câu hỏi hay làm theo lệnh.

**Mục tiêu của SFT:**
*   Dạy mô hình cách **hành xử** (Behavior) mong muốn.
*   Biến Base model -> Chat/Instruct model.

**Quy trình:**
*   Input: Các cặp câu hỏi - câu trả lời (Prompt - Response) chất lượng cao.
*   Loss: Vẫn dùng Next token prediction, nhưng chỉ tính loss trên phần câu trả lời (Response), không tính trên phần câu hỏi (Prompt).

---

## 2. Instruction Tuning
Là một dạng của SFT, tập trung vào việc dạy mô hình tuân theo các chỉ dẫn (instructions) đa dạng.

*   **Tác vụ:** Tóm tắt, dịch, viết code, giải toán, viết thơ, danh sách...
*   **Zero-shot Generalization:** Sau khi được Instruction Tuning trên nhiều tác vụ, mô hình có khả năng thực hiện cả những tác vụ mới mà nó chưa từng thấy trong quá trình training (nhờ khả năng suy luận tổng quát).

---

## 3. Dữ liệu cho SFT
Chất lượng quan trọng hơn số lượng ("Quality is King").

*   **Nguồn dữ liệu:**
    *   *Human-generated:* Do con người viết (đắt đỏ, chất lượng cao).
    *   *LLM-generated (Synthetic data):* Dùng mô hình mạnh (GPT-4) để tạo dữ liệu training cho mô hình nhỏ hơn. (Rẻ, nhanh, nhưng cần kiểm soát chất lượng).
*   **Quy mô:** Chỉ cần hàng nghìn đến hàng trăm nghìn mẫu (ít hơn nhiều so với hàng tỷ tokens của Pre-training).
*   **Ví dụ:** Dataset phổ biến: Alpaca, Vicuna, Lima.

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)
Fine-tuning toàn bộ mô hình (Full Fine-tuning) rất tốn kém (cần VRAM gấp nhiều lần kích thước mô hình để lưu Optimizer states).

**PEFT:** Chỉ cập nhật một phần nhỏ tham số hoặc thêm các module nhỏ vào mô hình, giữ đông cứng (freeze) phần lớn trọng số gốc.

**Lợi ích:**
*   Giảm yêu cầu VRAM.
*   Tránh hiện tượng "Catastrophic Forgetting" (Quên kiến thức cũ).
*   Dễ dàng chia sẻ các Adapter nhỏ (vài chục MB) thay vì cả mô hình GB.

---

## 5. LoRA (Low-Rank Adaptation)
Kỹ thuật PEFT phổ biến nhất hiện nay.

**Ý tưởng:**
Thay vì cập nhật trực tiếp ma trận trọng số $W$ (kích thước $d \times d$), ta cập nhật thông qua 2 ma trận nhỏ $A$ và $B$:
$$ W' = W + \Delta W = W + BA $$
Trong đó:
*   $B$: kích thước $d \times r$
*   $A$: kích thước $r \times d$
*   $r$ (rank): rất nhỏ (ví dụ 8, 16, 64) so với $d$.

**Kết quả:** Số lượng tham số cần train giảm hàng nghìn lần, nhưng hiệu quả tương đương Full Fine-tuning.

---

## 6. QLoRA (Quantized LoRA)
Kết hợp Quantization và LoRA để train mô hình lớn trên GPU nhỏ.

*   **4-bit NormalFloat (NF4):** Một kiểu dữ liệu mới tối ưu cho trọng số phân phối chuẩn của Neural Network.
*   **Double Quantization:** Quantize cả các hằng số quantization để tiết kiệm thêm bộ nhớ.
*   **Paged Optimizers:** Sử dụng CPU RAM để offload optimizer states khi GPU bị tràn bộ nhớ (OOM).

-> Cho phép train mô hình 65B parameters trên một GPU 48GB.

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
| 📌 **[Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️](aero_llm_05_tuning_peft.md)** | [Xem bài viết →](aero_llm_05_tuning_peft.md) |
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
