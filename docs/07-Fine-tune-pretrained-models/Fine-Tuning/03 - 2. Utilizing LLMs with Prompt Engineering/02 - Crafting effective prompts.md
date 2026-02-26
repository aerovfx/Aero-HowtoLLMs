# Thiết Kế Prompt Hiệu Quả

## Giới Thiệu

Hãy khám phá các mẫu phổ biến cho thiết kế prompt và các khía cạnh chính cần xem xét, đảm bảo tương tác của bạn với các mô hình AI này vừa hiệu quả vừa năng suất.

Hãy nghĩ về prompt engineering như việc chế tạo một chiếc chìa khóa mở khóa toàn bộ tiềm năng của một bộ não AI phức tạp.

## Tầm Quan Trọng Của Thiết Kế Prompt

Thiết kế prompt rất quan trọng vì nó ảnh hưởng trực tiếp đến chất lượng và mức độ liên quan của các phản hồi bạn nhận được từ một LLM.

### Nguyên Tắc Thiết Kế

Một prompt được thiết kế tốt nên:
- **Rõ ràng (Clear):** Dễ hiểu
- **Ngắn gọn (Concise):** Không dài dòng
- **Cụ thể (Specific):** Đưa ra hướng dẫn cụ thể

Nó nên cung cấp đủ ngữ cảnh để hướng dẫn phản hồi của mô hình, nhưng không quá nhiều để làm quá tải hoặc làm confuse mô hình.

### Mẫu Prompt Cụ Thể

Một số mô hình, đặc biệt là các mô hình nhỏ hơn như T5, có các mẫu prompt cụ thể được tối ưu hóa cho việc huấn luyện của chúng. Ví dụ, T5 chuyển đổi mọi tác vụ thành định dạng text-to-text, thường bắt đầu bằng một từ khóa báo hiệu loại tác vụ, như "translate", "summarize", hoặc "question" cho việc trả lời câu hỏi.

Cách tiêu chuẩn hóa này giúp mô hình nhanh chóng nhận ra một tác vụ và áp dụng quy trình và chiến lược phù hợp.

## Ba Mẫu Chính Cho Prompt Engineering

### 1. Few-shot Pattern

Mẫu này involve cung cấp một số ví dụ về tác vụ trước khi trình bày cho mô hình một instance mới để giải quyết.

**Ví dụ:** Nếu bạn đang dạy mô hình nhận dạng tên động vật trong văn bản, bạn có thể đưa ra các ví dụ với động vật được gắn nhãn trước khi yêu cầu nó nhận dạng động vật trong một câu mới.

**Ví dụ cụ thể:**
- Prompt: "The quick brown fox jumps over the lazy dog" → Đáp án: fox, dog
- Prompt thực: "A sheep and a wolf became unlikely friends" → Yêu cầu nhận dạng động vật

### 2. Cognitive Verifier Pattern

Mẫu này cực kỳ hữu ích khi bạn cần một cách tiếp cận đúng về một chủ đề và không chắc chắn liệu chúng ta có đang giải quyết tất cả các khía cạnh của nó hay không. Bằng cách sử dụng mẫu này, LLM tăng độ tin cậy của các đầu ra bằng cách kiểm tra thông tin cần thiết trước phản hồi cuối cùng.

**Ví dụ prompt:**
"Every time I ask a question, only ask me for additional information to clarify what I'm asking before providing a final answer."

### 3. Question Refinement Pattern

Mẫu này được sử dụng để yêu cầu LLM tinh chỉnh hoặc làm rõ một câu hỏi trước khi trả lời. LLM đặt thêm câu hỏi để có thêm thông tin hoặc ngữ cảnh, sau đó sử dụng để cung cấp câu trả lời chính xác hoặc liên quan hơn.

**Ví dụ prompt:**
"Every time I ask a question, ask me additional questions to clarify what I'm asking before you provide an answer."

## Kết Luận

Khi bạn thử nghiệm với các mẫu này, bạn sẽ thấy rằng cách bạn diễn đạt một prompt có thể thay đổi đáng kể kết quả bạn đạt được với một LLM. Hãy thử các cách tiếp cận khác nhau, tinh chỉnh prompts của bạn dựa trên các phản hồi, và liên tục học từ tương tác của họ.

---

*Nguồn: File subtitle 02 - Crafting effective prompts.vtt*
