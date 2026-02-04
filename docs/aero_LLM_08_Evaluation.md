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
