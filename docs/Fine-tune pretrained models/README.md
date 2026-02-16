
# 🗂 Chỉ Mục: Fine-tuning Pretrained Models

Chào mừng bạn đến với mục tài liệu về **Fine-tuning các mô hình ngôn ngữ tiền huấn luyện**. Dưới đây là danh sách các bài học và thử thách mã nguồn (Code Challenges) được sắp xếp theo trình tự.

---

### 📚 Danh Sách Các Bài Học

1.  **[Chương 01: Tinh chỉnh (Fine-tuning) có nghĩa là gì?](./aero_LLM_01_What does fine-tuning mean.md)**
    *   Giới thiệu về khái niệm và mục đích của việc tinh chỉnh mô hình.
2.  **[Chương 02: Tinh chỉnh mô hình GPT-2 tiền huấn luyện](./aero_LLM_02_Fine-tune a pretrained GPT2.md)**
    *   Hướng dẫn thực hành tinh chỉnh GPT-2 trên tác phẩm *Gulliver's Travels*.
3.  **[Thử thách 03: Tốc độ học của Gulliver](./aero_LLM_03CodeChallenge Gulliver's learning rates.md)**
    *   Đánh giá ảnh hưởng của Learning Rate đến hiệu suất mô hình.
4.  **[Chương 04: Quy trình sinh văn bản từ các mô hình tiền huấn luyện](./aero_LLM_04_On generating text from pretrained models.md)**
    *   Nghiên cứu cách mô hình sinh dữ liệu sau khi được huấn luyện.
5.  **[Thử thách 05: Tối đa hóa yếu tố X](./aero_LLM_05_CodeChallenge Maximize the X factor..md)**
    *   Sử dụng KL Divergence để tối ưu hóa việc sinh các token cụ thể.
6.  **[Chương 06: Alice và Edgar Allan Poe (với GPT-Neo)](./aero_LLM_06_Alice in Wonderland and Edgar Allen Poe (with GPT-neo).md)**
    *   Thực hành tinh chỉnh phong cách văn học với GPT-Neo.
7.  **[Thử thách 07: Định lượng sự tinh chỉnh Alice-Edgar](./aero_LLM_07_CodeChallenge Quantify the AliceEdgar fine-tuning.md)**
    *   Sử dụng BERT để đo lường mức độ thành công của việc chuyển đổi phong cách.
8.  **[Thử thách 08: Cuộc trò chuyện giữa Alice và Edgar](./aero_LLM_08_CodeChallenge A chat between Alice and Edgar.md)**
    *   Mô phỏng hội thoại giữa hai mô hình mang phong cách khác nhau.
9.  **[Chương 09: Tinh chỉnh từng phần bằng cách đóng băng trọng số Attention](./aero_LLM_09_Partial fine-tuning by freezing attention weights.md)**
    *   Chiến lược đóng băng (freezing) để tối ưu hóa tham số.
10. **[Thử thách 10: Tinh chỉnh và đóng băng có mục tiêu (Phần 1)](./aero_LLM_010_CodeChallenge Fine-tuning and targeted freezing (part 1).md)**
    *   Thực hành kỹ thuật đóng băng tham số trên BERT.
11. **[Thử thách 11: Tinh chỉnh và đóng băng có mục tiêu (Phần 2)](./aero_LLM_011_CodeChallenge Fine-tuning and targeted freezing (part 2).md)**
    *   Tiếp tục tối ưu hóa quy trình đóng băng để đạt hiệu suất cao hơn.
12. **[Chương 12: Tinh chỉnh hiệu quả tham số (PEFT)](./aero_LLM_012_Parameter-efficient fine-tuning (PEFT).md)**
    *   Tổng quan về các kỹ thuật LoRA, Adapter, Prefix Tuning.
13. **[Chương 13: Sử dụng CodeGen để hoàn thành mã nguồn](./aero_LLM_013_CodeGen for code completion.md)**
    *   Kiến trúc và ứng dụng của mô hình CodeGen trong lập trình.
14. **[Thử thách 14: Tinh chỉnh CodeGen cho toán giải tích](./aero_LLM_014_CodeChallenge Fine-tune codeGen for calculus.md)**
    *   Huấn luyện mô hình sinh mã Python để giải quyết các bài toán toán học.
15. **[Chương 15: Tinh chỉnh BERT cho bài toán phân loại](./aero_LLM_015_Fine-tuning BERT for classification.md)**
    *   Cấu trúc và quy trình fine-tuning BERT cho dữ liệu văn bản.
16. **[Thử thách 16: Phân tích cảm xúc IMDB bằng BERT](./aero_LLM_016_CodeChallenge IMDB sentiment analysis using BERT.en_US.md)**
    *   Thực hành phân loại cảm xúc tích cực/tiêu cực trên dữ liệu điện ảnh.
17. **[Chương 17: Cắt gradient và bộ điều chỉnh tốc độ học (Phần 1)](./aero_LLM_017_Gradient clipping and learning rate scheduler (part 1).en_US.md)**
    *   Kỹ thuật ổn định quá trình huấn luyện bằng Gradient Clipping.
18. **[Chương 18: Cắt gradient và bộ điều chỉnh tốc độ học (Phần 2)](./aero_LLM_018_Gradient clipping and learning rate scheduler (part 2).md)**
    *   Tìm hiểu sâu về Learning Rate Schedulers (Cosine, Linear).
19. **[Thử thách 19: Cắt, Đóng băng và Điều chỉnh BERT](./aero_LLM_019_CodeChallenge Clip, freeze, and schedule BERT.md)**
    *   Kết hợp các kỹ thuật để tinh chỉnh BERT đạt độ chính xác ~90%.
20. **[Chương 20: Lưu và tải các mô hình đã huấn luyện](./aero_LLM_020_Saving and loading trained models.md)**
    *   Quản lý tham số mô hình trong PyTorch và Hugging Face.
21. **[Chương 21: BERT phân loại Alice hay Edgar](./aero_LLM_021_BERT decides Alice or Edgar.md)**
    *   Ứng dụng BERT trong nghiên cứu phong cách văn học số.
22. **[Thử thách 22: Sự tiến hóa của Alice và Edgar (Phần 1)](./aero_LLM_022_CodeChallenge Evolution of Alice and Edgar (part 1).md)**
    *   Đồng tiến hóa hệ thống sinh và phân loại văn bản.
23. **[Thử thách 23: Sự tiến hóa của Alice và Edgar (Phần 2)](./aero_LLM_023_CodeChallenge Evolution of Alice and Edgar (part 2).md)**
    *   Đánh giá chất lượng mô hình sinh thông qua mô hình phân loại.

---
*Ghi chú: Các tài liệu được biên soạn nhằm phục vụ mục tiêu nghiên cứu và đào tạo về LLM.*
