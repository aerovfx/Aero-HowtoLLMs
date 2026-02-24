# Thấu Kính Logit (The Logit Lens): Soi Sáng Tư Duy Tầng Trung Gian Của Mô Hình Ngôn Ngữ

## Tóm tắt (Abstract)
Phương pháp Thấu kính Logit (Logit Lens) cung cấp một giải pháp đột phá để giải mã "hộp đen" của Large Language Models (LLMs) mà không can thiệp thay đổi kiến trúc mô hình. Thay vì đợi đến khi Transformer hoàn tất quy trình ở lớp cuối cùng, kỹ thuật này kết nối trực tiếp các tầng ẩn trung gian (Intermediate Hidden States) với Ma trận giải mã nhúng (Unembedding Matrix/Language Model Head). Bằng việc sử dụng GPT-2 Small, bài báo cáo này minh chứng sự khả thi của việc giải mã và dự đoán trực tiếp Token tương lai ở mọi giao điểm nội bộ, phác họa bản đồ tư duy tiến hóa của mô hình qua từng block.

---

## 1. Mở Đầu (Introduction)
Kiến trúc tiêu chuẩn của một mô hình Transformer (như GPT) bao gồm:
1. Tầng Embedding (Token + Positional).
2. Chuỗi các khối Transformer Blocks (nơi thực hiện self-attention và feed-forward).
3. Tầng Unembedding (Thường là một ma trận tuyến tính - LM Head) để phóng chiếu vector đầu ra vào không gian Từ vựng (Vocabulary space).

Một câu hỏi mang tính khiêu khích được đặt ra về mặt giải thích AI (Explainable AI): *"Điều gì sẽ xảy ra nếu ta cắt ngang mô hình ở Layer thứ 3 và ép nó xuất dự đoán ngay lập tức?"* 
Phương pháp **Logit Lens** được phát minh độc lập trên cộng đồng LessWrong trả lời đúng trọng tâm câu hỏi trên: Thay vì di chuyển tuần tự qua mạng, ta giả lập rút ngắn quy trình bằng cách đưa trạng thái ẩn tầng $L$ tạt ngang vào trực tiếp bộ phân giải từ vựng cuối cùng.

---

## 2. Nguyên Lý Cơ Sự (Methodology)

### 2.1. Bản Chất Hình Học Của Transformer
Đầu ra của bất kỳ một Transformer Block nào cũng bảo toàn kích thước không gian vector của Embeddings gốc (VD GPT-2 Small là 768 dimensions). Nhờ sự giới hạn đồng nhất đa chiều này, Ma trận giải mã cuối cùng (`model.lm_head.weight`) dễ dàng xem bất kỳ tầng trung gian nào là "vật liệu khả thi" để nhân ma trận. 

### 2.2. Trích Xuất Dữ Liệu
Quy trình giả lập được khởi tạo:
- Hàm Forward Pass lấy mẫu với tuỳ chọn `output_hidden_states=True`. 
- Bỏ qua Layer 0 (Embedding ban đầu) vì chưa chịu tác động học tập (transformation) nào.
- 12 ma trận Layers được trích xuất (GPT-2 Small) có dạng `[1, seq_length, 768]`.

### 2.3. Giải Mã Sớm (Early Decoding)
Sử dụng công thức chiếu vector:
$$ \text{Logits}_{L} = \text{Hidden\_States}_{L} \times \text{LM\_Head}^T $$
Từ đó, ta áp dụng hàm $\text{argmax}$ phân bổ qua $\text{Softmax}$ cho ma trận từ vựng 50,000 chiếu, tìm ra từ có xác suất cao nhất tại chính Layer lơ lửng đó.

---

## 3. Khảo Sát Đánh Giá (Analysis & Visualizations)

Trích dẫn mẫu dùng cho thí nghiệm: 
*“The way you do anything is the way you do everything”*.

Khảo sát được thực hiện song song để dự báo Token "kế tiếp":
- **Ở Layer đầu tiên (Layer 1):** Mô hình xuất ra toàn văn bản nhiễu (garbage) và hư từ ("the", "else"). Điều này minh chứng tại chặng đầu, mô hình chỉ mới loay hoay tổng hợp ngữ pháp cục bộ mà chưa mường tượng nổi chuỗi nghĩa dài.
- **Ở Layer thứ 3:** Khi đến đoạn tính toán từ “do”, model dự báo từ tiếp theo sẽ là “not” (mặc dù câu gốc là “anything”). “do not” là một chuỗi n-gram vô cùng hợp lệ về mặt ngữ nghĩa ngữ pháp nội bộ, chứng minh khả năng nhúng chéo khối từ bắt đầu xuất hiện.
- **Nhiệt Đồ Chuyển Màu (Heatmap Visualization):** Tính toán Softmax tại định dạng Ma trận tổng $12 \text{ layers} \times \text{seq\_len}$ vẽ lên bức rèm chuyển pha. Logit Lens chỉ ra: Không cần đợi đến layer 12, bản ngã cụm từ đôi khi "chín" sớm và được chốt hạ đúng từ Layer 7-8 ở những từ khóa quan trọng.

---

## 4. Kết Luận
Logit Lens không phải thuật toán "đọc tâm trí" (Mind-reading) như cách giới truyền thông thổi phồng. Đây là kỹ thuật chiếu ảnh vector nghịch đảo (Inverse Projection) cho thấy mạng lưới tính toán Token thay đổi liên tục. Sự chuyển mình từ dự đoán "ngữ pháp sơ cấp" ở tầng nông đến "ngữ nghĩa trừu tượng" ở tầng sâu mở ra một kho tàng khổng lồ cho việc sửa chữa sai lệch mô hình (hallucination fixes) từ tận lõi kiến trúc.

---

## Tài Liệu Tham Khảo (Citations)
1. Thực nghiệm mô phỏng dựa theo lý thuyết từ bài báo "Logit Lens" trên diễn đàn LessWrong, ứng dụng vào GPT-2 bằng PyTorch rút trích tại `aero_LLM_15_The Logit Lens.md`.
