
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
# Lecture 7: Agentic LLMs & Tool Use 🛠️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này giới thiệu cách mở rộng khả năng của LLM thông qua việc sử dụng công cụ (Tool Use), kết nối với dữ liệu ngoài (RAG) và xây dựng các tác nhân tự chủ (Agents).

---

## 📚 Mục Lục
1. [Giới hạn của LLM & Giải pháp](#1-giới-hạn-của-llm--giải-pháp)
2. [RAG (Retrieval-Augmented Generation)](#2-18_rag-retrieval-augmented-generation)
3. [Tool Calling (Function Calling)](#3-tool-calling-function-calling)
4. [Agents (Tác nhân AI)](#4-agents-tác-nhân-ai)
5. [ReAct Framework](#5-react-framework)
6. [Multi-Agent Systems & MCP](#6-multi-agent-systems--mcp)

---

## 1. Giới hạn của LLM & Giải pháp
Mặc dù LLM rất mạnh, chúng vẫn có 3 điểm yếu lớn:
1.  **Kiến thức tĩnh (Static Knowledge):** Không biết thông tin mới sau ngày cắt dữ liệu (knowledge cutoff).
2.  **Ảo giác (Hallucination):** Tự bịa đặt thông tin khi không biết câu trả lời.
3.  **Không hành động (No Action):** Chỉ tạo ra văn bản, không thể tương tác với thế giới thực (gửi mail, đặt hàng).

-> **Giải pháp:** Kết nối LLM với công cụ và dữ liệu bên ngoài.

---

## 2. RAG (Retrieval-Augmented Generation)
Kỹ thuật giúp LLM truy cập dữ liệu mới mà không cần train lại.

**Quy trình 3 bước:**
1.  **Retrieve (Truy xuất):** Tìm kiếm các tài liệu liên quan từ Knowledge Base (dựa trên Vector Search/Semantic Search).
2.  **Augment (Bổ sung):** Đưa thông tin tìm được vào Prompt (Context).
3.  **Generate (Sinh văn bản):** LLM trả lời câu hỏi dựa trên thông tin được cung cấp.

**Kỹ thuật nâng cao:**
*   **Chunking:** Chia nhỏ văn bản thành các đoạn (chunks) vừa vặn (khoảng 500 tokens).
*   **Hybrid Search:** Kết hợp Vector Search (Semantic) và Keyword Search (BM25) để tăng độ chính xác.
*   **Re-ranking:** Dùng mô hình Cross-Encoder để sắp xếp lại kết quả tìm kiếm cho chính xác hơn trước khi đưa vào LLM.

---

## 3. Tool Calling (Function Calling)
Cho phép LLM sử dụng các công cụ bên ngoài (Calculator, Weather API, Database...).

**Cơ chế:**
1.  **Định nghĩa:** Người lập trình cung cấp mô tả công cụ (Tên, Tham số, Công dụng) cho LLM.
2.  **Quyết định:** LLM quyết định xem có cần dùng công cụ không. Nếu cần, nó sinh ra một cấu trúc JSON chứa tên hàm và tham số.
3.  **Thực thi:** Hệ thống thực thi hàm đó và trả kết quả về cho LLM.
4.  **Trả lời:** LLM dùng kết quả đó để trả lời người dùng.

---

## 4. Agents (Tác nhân AI)
Agent là một hệ thống dùng LLM làm "bộ não" để tự chủ giải quyết vấn đề qua nhiều bước.
*   **Khác biệt với Tool Calling:** Tool Calling chỉ là một bước đơn lẻ. Agent có khả năng lập kế hoạch (Plan), ghi nhớ (Memory) và tự sửa lỗi (Self-correction).

---

## 5. ReAct Framework
Phương pháp phổ biến để xây dựng Agent: **Re**ason + **Act**.

**Vòng lặp ReAct:**
1.  **Thought (Suy nghĩ):** Phân tích vấn đề, lập kế hoạch. ("Người dùng thấy lạnh -> Cần kiểm tra nhiệt độ phòng").
2.  **Action (Hành động):** Gọi công cụ. (`get_temperature()`).
3.  **Observation (Quan sát):** Nhận kết quả từ công cụ. ("Nhiệt độ là 18 độ C").
4.  **Thought (Suy nghĩ tiếp):** ("18 độ là lạnh -> Cần bật điều hòa").
5.  ... Lặp lại cho đến khi xong việc.

---

## 6. Multi-Agent Systems & MCP
*   **Multi-Agent:** Thay vì một Agent làm tất cả, ta dùng nhiều Agent chuyên biệt (Coder, Writer, Reviewer) phối hợp với nhau.
*   **MCP (Model Context Protocol):** Tiêu chuẩn mới (từ Anthropic) giúp chuẩn hóa cách kết nối LLM với các nguồn dữ liệu và công cụ, giúp tránh việc phải viết lại code kết nối cho từng mô hình/ứng dụng khác nhau.

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
| [Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️](aero_llm_05_tuning_peft.md) | [Xem bài viết →](aero_llm_05_tuning_peft.md) |
| [Lecture 6: LLM Reasoning 🧠](aero_llm_06_reasoning.md) | [Xem bài viết →](aero_llm_06_reasoning.md) |
| 📌 **[Lecture 7: Agentic LLMs & Tool Use 🛠️](aero_llm_07_agentic_llms.md)** | [Xem bài viết →](aero_llm_07_agentic_llms.md) |
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
