
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [09 Quantitative evaluations](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Độ Chính Xác, Tính Mạch Lạc và Sự Phù Hợp trong Đánh Giá Mô Hình Ngôn Ngữ

## Tóm tắt

Có những đặc tính của văn bản rất khó đánh giá bằng các bài kiểm tra tiêu chuẩn hoặc bằng các chỉ số logit đơn lẻ. Các thuộc tính như tính mạch lạc (coherence), sự phù hợp (relevance), giọng văn (tone), và sự thân thiện đều mang tính chủ quan cao. Bài viết này thảo luận về các phương pháp đánh giá định tính dựa trên sự phản hồi của con người (Human Feedback) và xu hướng sử dụng các mô hình ngôn ngữ tiên phong (Frontier Models) để tự động hóa quá trình này.

---

## 1. Bản chất Chủ quan của Ngôn ngữ Tự nhiên

Ngôn ngữ không chỉ là sự kết hợp của các từ ngữ đúng ngữ pháp mà còn chứa đựng cảm xúc và ngữ cảnh. Các chỉ số định lượng như *Perplexity* có thể đo mức độ dự đoán chính xác token tiếp theo, nhưng không thể đo được:
- **Tính mạch lạc (Coherence):** Sự kết nối logic giữa các ý tưởng trong một đoạn văn dài.
- **Sự phù hợp (Relevance):** Câu trả lời có bám sát nhu cầu thực tế của người dùng hay không.
- **Phong cách (Style/Texture):** Ví dụ như sự khác biệt giữa văn phong viết tay chân thực và phong cách "LinkedIn" (câu ngắn, nhiều emoji, punchy) thường thấy ở nội dung do AI tạo ra.

---

## 2. Đánh giá bởi Con người (Human Evaluation)

Giải pháp cho vấn đề chủ quan là đưa con người vào vòng lặp đánh giá (Human-in-the-loop). Các phương pháp phổ biến bao gồm:

### 2.1 Thang đo Điểm số (Numeric Feedback)
Người đánh giá cho điểm văn bản trên thang từ 1-10 dựa trên các tiêu chí cụ thể (ví dụ: độ hữu ích).

### 2.2 So sánh Cặp (A/B Testing)
Người đánh giá chọn văn bản tốt hơn trong hai lựa chọn được đưa ra. Đây là cơ sở cho các bảng xếp hạng như *LMSYS Chatbot Arena*.

### 2.3 Kiểm tra Turing (Turing-like tests)
Thử thách người đánh giá phân biệt đâu là văn bản do con người viết và đâu là do AI tạo ra để đo lường độ "tự nhiên".

---

## 3. RLHF: Reinforcement Learning from Human Feedback

Kết quả từ các đánh giá này được sử dụng để huấn luyện một "Reward Model" (Mô hình phần thưởng), sau đó được dùng để tối ưu hóa LLM thông qua thuật toán PPO (Proximal Policy Optimization).

Mục tiêu là tối ưu hóa hàm giá trị:

$$J(\phi) = \mathbb{E}_{x \sim D, y \sim \pi_{\phi}(y|x)} [R(x, y)] - \beta \text{KL}(\pi_{\phi} || \pi_{ref})$$

Trong đó:
- $R(x, y)$ là điểm số từ Reward Model phản ánh sở thích của con người.
- $\text{KL}$ là độ lệch Kullback-Leibler để đảm bảo mô hình không đi quá xa so với mô hình tham chiếu ban đầu.

---

## 4. Xu hướng Tự động hóa bằng LLM

Do chi phí thuê chuyên gia đánh giá là rất lớn, các nhà phát triển đang chuyển sang sử dụng các mô hình mạnh nhất (như GPT-4) để đánh giá các mô hình nhỏ hơn. 
- **Ưu điểm:** Tốc độ nhanh, khả năng mở rộng tốt.
- **Hạn chế:** Nguy cơ gặp phải "vòng lặp phản hồi" khiến mô hình mất đi tính tự nhiên của con người và chỉ tối ưu hóa theo những tiêu chí cứng nhắc của AI.

---

## Tài liệu tham khảo

1. **Ouyang, L., et al. (2022).** *Training language models to follow instructions with human feedback.* (InstructGPT paper).
2. **Stiennon, N., et al. (2020).** *Learning to summarize with human feedback.*
3. **LMSYS Org.** *Chatbot Arena: Benchmarking LLMs in the Wild.*
4. **Ziegler, D. M., et al. (2019).** *Fine-Tuning Language Models from Human Preferences.*
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
