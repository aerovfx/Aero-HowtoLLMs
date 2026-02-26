# 📂 Module: 07-Fine-tune-pretrained-models
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]() [![Content: 100% Vietnamese](https://img.shields.io/badge/Content-Vietnamese-red.svg)]()

[Home](../README.md) > **07-Fine-tune-pretrained-models**

---
### 🧭 Quick Navigation

- [🏠 Cổng tài liệu](../README.md)
- [📚 Module 01: LLM Course](../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../19-AI-safety/index.md)
---


Chào mừng bạn đến với mục tài liệu về **Fine-tuning (Tinh chỉnh) các mô hình ngôn ngữ tiền huấn luyện**. Thư mục này chứa lộ trình thực chiến từ cơ bản đến nâng cao, tập trung vào việc tùy biến mô hình cho các bài toán chuyên biệt.

---

### 📚 Lộ Trình Học Tập (23 Tài Liệu)

#### 🔹 Phần 1: Nền Tảng & GPT-2 (Basic Fine-tuning)
1.  **[Chương 01: Khái niệm về Fine-tuning](./aero_LLM_01_What does fine-tuning mean.md)** - Tại sao và khi nào cần tinh chỉnh?
2.  **[Chương 02: Thực hành Fine-tune GPT-2](./aero_LLM_02_Fine-tune a pretrained GPT2.md)** - Tinh chỉnh trên tác phẩm *Gulliver's Travels*.
3.  **[Thử thách 03: Tối ưu Learning Rate](./aero_LLM_03CodeChallenge Gulliver's learning rates.md)** - Phân tích tốc độ học cho dữ liệu văn học.
4.  **[Chương 04: Cơ chế sinh văn bản](./aero_LLM_04_On generating text from pretrained models.md)** - Cách mô hình dự đoán token tiếp theo.
5.  **[Thử thách 05: Hàm mất mát KL Divergence](./aero_LLM_05_CodeChallenge Maximize the X factor..md)** - Tối ưu hóa việc sinh các ký tự mục tiêu (Yếu tố X).

#### 🔹 Phần 2: Series Alice & Edgar (Style Mimicry)
6.  **[Chương 06: Fine-tune phong cách với GPT-Neo](./aero_LLM_06_Alice in Wonderland and Edgar Allen Poe (with GPT-neo).md)** - Kết hợp Lewis Carroll và Edgar Allan Poe.
7.  **[Thử thách 07: Định lượng hiệu quả tinh chỉnh](./aero_LLM_07_CodeChallenge Quantify the AliceEdgar fine-tuning.md)** - Sử dụng mô hình phân loại để đo lường.
8.  **[Thử thách 08: Mô phỏng hội thoại đa mô hình](./aero_LLM_08_CodeChallenge A chat between Alice and Edgar.md)** - Cho Alice "trò chuyện" với Edgar.
9.  **[Chương 09: Chiến lược Đóng băng Attention](./aero_LLM_09_Partial fine-tuning by freezing attention weights.md)** - Tinh chỉnh từng phần để tiết kiệm tài nguyên.

#### 🔹 Phần 3: Kỹ thuật Tối ưu & Tốc độ (Advanced Tuning)
10. **[Thử thách 10: Targeted Freezing (Phần 1)](./aero_LLM_010_CodeChallenge Fine-tuning and targeted freezing (part 1).md)** - Đóng băng lớp có chọn lọc.
11. **[Thử thách 11: Targeted Freezing (Phần 2)](./aero_LLM_011_CodeChallenge Fine-tuning and targeted freezing (part 2).md)** - Nâng cao hiệu suất đóng băng.
12. **[Chương 12: Tổng quan về PEFT](./aero_LLM_012_Parameter-efficient fine-tuning (PEFT).md)** - LoRA, Adapters và các kỹ thuật mới.
13. **[Chương 13: Mô hình CodeGen](./aero_LLM_013_CodeGen for code completion.md)** - Fine-tuning dành riêng cho lập trình.
14. **[Thử thách 14: Sinh mã cho toán giải tích](./aero_LLM_014_CodeChallenge Fine-tune codeGen for calculus.md)** - Ứng dụng CodeGen trong toán học.

#### 🔹 Phần 4: Phân Loại & Ổn Định (Classification & Stability)
15. **[Chương 15: Fine-tuning BERT phân loại](./aero_LLM_015_Fine-tuning BERT for classification.md)** - Chuyển đổi mô hình sinh sang mô hình phân loại.
16. **[Thử thách 16: Phân tích cảm xúc IMDB](./aero_LLM_016_CodeChallenge IMDB sentiment analysis using BERT.en_US.md)** - Đánh giá review phim bằng BERT.
17. **[Chương 17: Gradient Clipping (Phần 1)](./aero_LLM_017_Gradient clipping and learning rate scheduler (part 1).en_US.md)** - Chống bùng nổ gradient.
18. **[Chương 18: Gradient Clipping (Phần 2)](./aero_LLM_018_Gradient clipping and learning rate scheduler (part 2).md)** - Sử dụng Scheduler để điều phối LR.
19. **[Thử thách 19: Quy trình Clip, Freeze & Schedule](./aero_LLM_019_CodeChallenge Clip, freeze, and schedule BERT.md)** - Kết hợp bộ ba kỹ thuật tối ưu.

#### 🔹 Phần 5: Triển Khai & Đánh Giá (Deployment & Evaluation)
20. **[Chương 20: Quản lý tham số & Lưu trữ](./aero_LLM_020_Saving and loading trained models.md)** - Lưu/Tải checkpoint trong PyTorch.
21. **[Chương 21: BERT - Trọng tài văn học](./aero_LLM_021_BERT decides Alice or Edgar.md)** - Sử dụng BERT để phân loại tác giả.
22. **[Thử thách 22: Tiến hóa hệ thống (Phần 1)](./aero_LLM_022_CodeChallenge Evolution of Alice and Edgar (part 1).md)** - Quy trình cập nhật mô hình liên tục.
23. **[Thử thách 23: Tiến hóa hệ thống (Phần 2)](./aero_LLM_023_CodeChallenge Evolution of Alice and Edgar (part 2).md)** - Đánh giá trung gian và kết luận.

---

### 🛠️ Yêu Cầu Thực Hành
- Các ví dụ mã nguồn sử dụng thư viện **Transformers (Hugging Face)** và **PyTorch**.
- Nên sử dụng GPU (T4 trở lên) để chạy các thử thách về BERT và GPT-Neo.

---
*Biên soạn phục vụ dự án Aero-HowtoLLMs.*

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*