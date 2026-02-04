# Lecture 7: Agentic LLMs & Tool Use 🛠️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này giới thiệu cách mở rộng khả năng của LLM thông qua việc sử dụng công cụ (Tool Use), kết nối với dữ liệu ngoài (RAG) và xây dựng các tác nhân tự chủ (Agents).

---

## 📚 Mục Lục
1. [Giới hạn của LLM & Giải pháp](#1-giới-hạn-của-llm--giải-pháp)
2. [RAG (Retrieval-Augmented Generation)](#2-rag-retrieval-augmented-generation)
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
