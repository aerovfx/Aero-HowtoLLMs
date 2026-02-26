
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
# Lecture 1: Transformer Architecture 🤖

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này giới thiệu nền tảng của các mô hình ngôn ngữ hiện đại: Kiến trúc Transformer và cơ chế Self-Attention.

---

## 📚 Mục Lục
1. [Giới thiệu về NLP](#1-giới-thiệu-về-nlp)
2. [Tokenization (Mã hóa văn bản)](#2-tokenization-mã-hóa-văn-bản)
3. [Word Embeddings (Biểu diễn từ)](#3-word-embeddings-biểu-diễn-từ)
4. [Sự hạn chế của RNN/LSTM](#4-sự-hạn-chế-của-rnnlstm)
5. [Cơ chế Attention & Self-Attention](#5-cơ-chế-attention--self-attention)
6. [Kiến trúc Transformer](#6-kiến-trúc-transformer)

---

## 1. Giới thiệu về NLP
**Natural Language Processing (NLP)** là lĩnh vực xử lý và tính toán trên văn bản. Có 3 nhóm tác vụ chính:
*   **Classification (Phân loại):** Đầu vào là text, đầu ra là nhãn (VD: Sentiment Analysis - Phân tích cảm xúc).
*   **Multi-classification / Tagging:** Gán nhãn cho từng từ hoặc thực thể (VD: Named Entity Recognition - NER).
*   **Generation (Sinh văn bản):** Đầu vào là text, đầu ra là text (VD: Dịch máy, Chatbot). Đây là nhóm tác vụ phổ biến nhất hiện nay với LLMs.

---

## 2. Tokenization (Mã hóa văn bản)
Mô hình không hiểu văn bản thô, nó cần các con số. Quá trình chia nhỏ văn bản thành các đơn vị cơ bản gọi là **Tokenization**.

*   **Word-level (Mức từ):** Chia theo dấu cách.
    *   *Nhược điểm:* Không xử lý được từ chưa biết (OOV - Out Of Vocabulary), không tận dụng được gốc từ (root words).
*   **Character-level (Mức ký tự):** Chia theo từng chữ cái.
    *   *Nhược điểm:* Chuỗi quá dài, mất ngữ nghĩa, tính toán chậm.
*   **Subword-level (Mức từ con - Phổ biến nhất):** Chia từ thành các phần nhỏ hơn có nghĩa (VD: "reading" -> "read" + "ing").
    *   *Ưu điểm:* Cân bằng giữa độ dài chuỗi và kích thước từ điển, xử lý tốt từ hiếm.

---

## 3. Word Embeddings (Biểu diễn từ)
Sau khi có token (ID), ta cần chuyển nó thành vector số học gọi là **Embedding**.
*   **One-hot Encoding:** Vector toàn số 0 và một số 1. *Nhược điểm:* Không thể hiện được sự tương đồng giữa các từ (các vector đều trực giao).
*   **Learned Embeddings (Word2Vec):** Học biểu diễn từ sao cho các từ có ngữ nghĩa giống nhau sẽ nằm gần nhau trong không gian vector (VD: King - Man + Woman ≈ Queen).

---

## 4. Sự hạn chế của RNN/LSTM
Trước Transformer, RNN (Recurrent Neural Networks) và LSTM là chuẩn mực.
*   **Cơ chế:** Xử lý tuần tự từng từ một (word by word), giữ lại "bộ nhớ" (hidden state) về các từ đã qua.
*   **Nhược điểm chí tử:**
    *   **Long-range dependency (Phụ thuộc xa):** Khó nhớ được thông tin từ đầu câu khi đã đi đến cuối câu (vấn đề Vanishing Gradient).
    *   **Không thể song song hóa:** Phải đợi từ trước xử lý xong mới đến từ sau -> Tốc độ huấn luyện rất chậm.

---

## 5. Cơ chế Attention & Self-Attention
Để giải quyết vấn đề "quên" của RNN, cơ chế **Attention** ra đời (2014) và đỉnh cao là **Self-Attention** (2017).

**Ý tưởng:** Thay vì xử lý tuần tự, hãy cho phép mỗi từ "nhìn" thấy tất cả các từ khác trong câu cùng một lúc và tự quyết định xem từ nào quan trọng với mình.

### Công thức Self-Attention
Mỗi token được chiếu thành 3 vector: **Query (Q)**, **Key (K)**, **Value (V)**.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

*   **Q (Truy vấn):** Tôi đang tìm kiếm thông tin gì?
*   **K (Chìa khóa):** Tôi có thông tin gì để cung cấp?
*   **V (Giá trị):** Nội dung thông tin của tôi là gì?
*   **$QK^T$:** Tính độ tương đồng (score) giữa truy vấn và chìa khóa.
*   **Softmax:** Chuẩn hóa score thành trọng số (tổng = 1).
*   **Nhân với V:** Tổng hợp thông tin từ các từ quan trọng.

---

## 6. Kiến trúc Transformer
Mô hình **Transformer** (trong bài báo "Attention Is All You Need") bao gồm:

### Encoder (Bộ mã hóa)
*   Xử lý chuỗi đầu vào (Input).
*   Dùng **Self-Attention** để hiểu ngữ cảnh của từ trong câu.
*   Đầu ra: Các vector embedding giàu ngữ nghĩa.

### Decoder (Bộ giải mã)
*   Sinh chuỗi đầu ra (Output).
*   Dùng **Masked Self-Attention** (chỉ nhìn thấy các từ phía trước, không nhìn thấy tương lai).
*   Dùng **Cross-Attention** để lấy thông tin từ Encoder.

### Các thành phần khác
*   **Positional Encoding:** Vì Transformer xử lý song song nên không biết thứ tự từ -> Cần cộng thêm vector vị trí để báo cho mô hình biết từ nào đứng trước/sau.
*   **Multi-Head Attention:** Thay vì chỉ có 1 bộ Q,K,V, ta dùng nhiều bộ (Heads) để mô hình có thể học nhiều mối quan hệ khác nhau cùng lúc.
*   **Feed Forward Network (FFN):** Mạng nơ-ron truyền thẳng để xử lý thông tin sau lớp Attention.
*   **Add & Norm:** Cộng phần dư (Residual connection) và chuẩn hóa lớp (LayerNorm) để huấn luyện ổn định hơn.

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠](aero_llm_00_overview.md) | [Xem bài viết →](aero_llm_00_overview.md) |
| 📌 **[Lecture 1: Transformer Architecture 🤖](aero_llm_01_transformer.md)** | [Xem bài viết →](aero_llm_01_transformer.md) |
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
