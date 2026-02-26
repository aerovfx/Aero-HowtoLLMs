
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
# Lecture 8: LLM Evaluation ⚖️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này tập trung vào bài toán khó nhất trong phát triển LLM: Làm sao để đánh giá (Evaluate) mô hình một cách chính xác và tin cậy?

---

## 📚 Mục Lục
1. [Tại sao đánh giá LLM lại khó?](#1-tại-sao-đánh-giá-llm-lại-khó)
2. [Các phương pháp đánh giá](#2-các-phương-pháp-đánh-giá)
3. [Metrics truyền thống (BLEU, ROUGE)](#3-metrics-truyền-thống-bleu-rouge)
4. [LLM-as-a-Judge](#4-llm-as-a-judge)
5. [Benchmarks phổ biến](#5-benchmarks-phổ-biến)

---

## 1. Tại sao đánh giá LLM lại khó?
Khác với các bài toán ML truyền thống (Classification, Regression) có đáp án đúng/sai rõ ràng, đầu ra của LLM là **Free-form Text (Văn bản tự do)**.
*   **Subjectivity (Tính chủ quan):** Một câu trả lời có thể hay với người này nhưng dở với người kia.
*   **Variety (Sự đa dạng):** Có vô số cách để diễn đạt cùng một ý.
*   **Chi phí:** Đánh giá thủ công bởi con người (Human Eval) rất đắt và chậm.

---

## 2. Các phương pháp đánh giá
1.  **Human Evaluation:** Chính xác nhất nhưng tốn kém nhất. Dùng cho giai đoạn cuối hoặc kiểm tra ngẫu nhiên (Spot check).
2.  **Code Evaluation:** Dùng Unit Test để chấm điểm code do LLM sinh ra (Pass@k). Rất chính xác cho bài toán lập trình.
3.  **Algorithmic Metrics:** Dùng công thức toán học để so sánh với văn bản mẫu (Reference).
4.  **Model-based Evaluation (LLM-as-a-Judge):** Dùng một LLM mạnh hơn (ví dụ GPT-4) để chấm điểm LLM yếu hơn.

---

## 3. Metrics truyền thống (BLEU, ROUGE)
Xuất phát từ dịch máy và tóm tắt văn bản.
*   **BLEU (Bilingual Evaluation Understudy):** Đếm số từ (n-grams) trùng lặp giữa câu dự đoán và câu mẫu. Chú trọng độ chính xác (Precision).
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Tương tự BLEU nhưng chú trọng độ bao phủ (Recall). Thường dùng cho tóm tắt.
*   **Nhược điểm:** Chỉ bắt lỗi chính tả/từ ngữ, không hiểu ngữ nghĩa. (Ví dụ: "I love you" và "I adore you" có ý nghĩa giống nhau nhưng điểm BLEU sẽ thấp vì không trùng từ). -> *Không còn phù hợp cho LLM hiện đại.*

---

## 4. LLM-as-a-Judge 👨‍⚖️
Phương pháp phổ biến nhất hiện nay để đánh giá Chatbot.

**Cơ chế:**
*   Đưa Prompt + Response của mô hình cần chấm + Reference (nếu có) + Tiêu chí chấm (Rubric) cho GPT-4 (hoặc Claude 3.5).
*   Yêu cầu GPT-4 đóng vai giám khảo, chấm điểm (thang 1-5 hoặc 1-10) và đưa ra lời giải thích (Rationale).

**Ưu điểm:**
*   Nhanh, rẻ, scale tốt.
*   Hiểu được ngữ nghĩa và sắc thái.
*   Độ tương quan (Correlation) cao với đánh giá của con người.

**Nhược điểm:**
*   **Position Bias:** Thường ưu tiên câu trả lời xuất hiện trước (hoặc sau).
*   **Verbosity Bias:** Thích câu trả lời dài dòng hơn.
*   **Self-preference Bias:** Thích văn phong giống chính nó.

---

## 5. Benchmarks phổ biến
Các bộ đề thi tiêu chuẩn để so sánh các mô hình:
*   **MMLU (Massive Multitask Language Understanding):** Kiến thức tổng quát (Toán, Lý, Hóa, Sử...).
*   **GSM8K:** Toán tiểu học (cần suy luận nhiều bước).
*   **HumanEval / MBPP:** Lập trình Python.
*   **Chatbot Arena (LMSYS):** Bảng xếp hạng dựa trên bình chọn mù (Blind test) của cộng đồng người dùng thực tế (Elo rating). *Đây được coi là thước đo uy tín nhất hiện nay.*

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [CS229: Xây Dựng Mô Hình Ngôn Ngữ Lớn (LLMs) 🧠](aero_LLM_00_Overview.md) | [Xem bài viết →](aero_LLM_00_Overview.md) |
| [Lecture 1: Transformer Architecture 🤖](aero_LLM_01_Transformer.md) | [Xem bài viết →](aero_LLM_01_Transformer.md) |
| [Lecture 2: Transformer Tricks & BERT 🛠️](aero_LLM_02_Transformer_Tricks.md) | [Xem bài viết →](aero_LLM_02_Transformer_Tricks.md) |
| [Lecture 3: Large Language Models (LLMs) & Inference 🚀](aero_LLM_03_Large_Language_Models.md) | [Xem bài viết →](aero_LLM_03_Large_Language_Models.md) |
| [Lecture 4: LLM Training - Pre-training 🏋️](aero_LLM_04_Training_Pretraining.md) | [Xem bài viết →](aero_LLM_04_Training_Pretraining.md) |
| [Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️](aero_LLM_05_Tuning_PEFT.md) | [Xem bài viết →](aero_LLM_05_Tuning_PEFT.md) |
| [Lecture 6: LLM Reasoning 🧠](aero_LLM_06_Reasoning.md) | [Xem bài viết →](aero_LLM_06_Reasoning.md) |
| [Lecture 7: Agentic LLMs & Tool Use 🛠️](aero_LLM_07_Agentic_LLMs.md) | [Xem bài viết →](aero_LLM_07_Agentic_LLMs.md) |
| 📌 **[Lecture 8: LLM Evaluation ⚖️](aero_LLM_08_Evaluation.md)** | [Xem bài viết →](aero_LLM_08_Evaluation.md) |
| [Lecture 9: Recap & Current Trends 🔮](aero_LLM_09_Trends.md) | [Xem bài viết →](aero_LLM_09_Trends.md) |
| [🛠️ Top 12 Repo Quan Trọng Cho AI Engineer Tối Ưu LLM](aero_LLM_10_Essential_Tools.md) | [Xem bài viết →](aero_LLM_10_Essential_Tools.md) |
| [Chương 1: Tổng Quan Về Large Language Models (LLMs) 🧠](aero_LLM_chapter01_overview_detailed.md) | [Xem bài viết →](aero_LLM_chapter01_overview_detailed.md) |
| [Chương 2: 5 Trụ Cột Của Việc Huấn Luyện LLMs 🏛️](aero_LLM_chapter02_5pillars_part1.md) | [Xem bài viết →](aero_LLM_chapter02_5pillars_part1.md) |
| [Chương 2: 5 Trụ Cột - Part 2 (Evaluation & Systems)](aero_LLM_chapter02_5pillars_part2.md) | [Xem bài viết →](aero_LLM_chapter02_5pillars_part2.md) |
| [Chương 3: Pre-training → Post-training Pipeline 🔄](aero_LLM_chapter03_training_pipeline.md) | [Xem bài viết →](aero_LLM_chapter03_training_pipeline.md) |
| [Chương 4 & 5: Mechanisms & Evaluation 🔧📊](aero_LLM_chapter04_05_mechanisms_eval.md) | [Xem bài viết →](aero_LLM_chapter04_05_mechanisms_eval.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
