# Thử Thách Lập Trình: So Sánh Độ Dài Quỹ Đạo Của Danh Từ Và Tính Từ (Phần 1)

## Tóm tắt (Abstract)
Thực nghiệm này tích hợp thư viện từ loại `spaCy` vào phân tích Quỹ đạo không gian trạng thái (State-space trajectories) để đo lường mức độ biến thiên của luồng ẩn (Hidden States) giữa hai loại từ trọng tâm: Danh từ (Nouns) và Tính từ (Adjectives). Thông qua việc thiết lập bộ quy tắc đồng bộ hóa mã thông báo (Token Synchronization) chống nhiễu BPE trên văn bản thực (Tiểu thuyết Frankenstein), thuật toán trích xuất thành công 300 mẫu tokens nguyên vẹn hợp lệ. Chỉ số Scree Plot PCA trên dữ liệu thực tế cho thấy sự bùng nổ phương sai phức tạp, phủ nhận kết quả nén 80% từ những câu văn tĩnh trước đó, phơi bày giới hạn tỷ lệ bảo toàn dữ liệu (<40%) của PCA khi đối mặt với văn học tự nhiên mở rộng.

---

## 1. Mở Đầu (Introduction)
Từ loại không thể tự giải nghĩa nếu bị tước đoạt khỏi context của nó. Tính từ (Adjectives) thường phụ thuộc vào việc điều chỉnh (modify) các danh từ gần kề, trong khi Danh từ (Nouns) mang ý nghĩa kết cấu hệ thống liên kết chủ thể - khách thể và hệ thống động từ trong toàn câu. 
Câu hỏi nghiên cứu: Giữa nhóm từ chuyên đóng vai trò kiến tạo cấu trúc ngữ pháp (Danh từ) và nhóm từ mang thiên hướng mô tả biên độ (Tính từ), loại nào sẽ kích hoạt biên độ luân chuyển lớn hơn, tạo ra quỹ đạo Vector PCA trải dài hơn khi chảy dọc theo 36 Transformer Layers của GPT-2 Large?

---

## 2. Tiết Thiết Lập (Methodology): Kỹ Thuật Dung Hòa Tokenizer

### 2.1. Giải Mã Sự Lạm Phát Token (Token Inflation)
Sử dụng nguyên tác "Frankenstein" từ Project Gutenberg (bỏ qua 1000 tokens đầu là rác meta-data). Nếu băm văn bản bằng 2 hệ Tokenizer: `spaCy` (Quy tắc ngôn từ nguyên thể) và `GPT-2` (Thuật toán Sub-words BPE), kết quả lượng Token của mô hình GPT luôn nhiều hơn khoảng $20\%$ so với `spaCy`. Sự lạm phát này gây ra hiện tượng so le Index nghiêm trọng.

### 2.2. Bộ Lọc Từ Nguyên Vẹn (Whole-Word Filter Strategy)
Để tránh nhãn sai (Mislabeling) khi GPT-2 cắt vụn từ thành subwords làm rối loạn cơ chế dán nhãn của `spaCy`, thực nghiệm đưa ra bộ lọc hà khắc để vứt bỏ rủi ro, hy sinh số lượng để lấy chất lượng chuẩn xác tuyệt đối:
- **Độ dài ký tự (Length):** >= 5 ký tự.
- **Tiêu chí Không gian (Whitespace heuristic):** Khẳng định token đang xét CÓ dấu cách làm ký tự mở đầu (tức không nối đuôi một chữ khác), VÀ Token kế tiếp ngay sau nó CŨNG CÓ dấu cách mở đầu (tức là Token này kết thúc trọn vẹn chứ không bị băm dư thừa hậu tố).
Điều kiện này đào thải triệt để mọi từ ghép, từ ngắt đoạn, từ sát dấu câu, giúp cất mẻ cào được đúng 150 mẫu Danh Từ và 150 mẫu Tính Từ (kèm theo Context Window 40 từ trước và 10 từ sau mỗi mốc Target). 

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Truy Xuất Không Gian Đa Chiều (Multidimensional Extraction)
Bơm 300 mẫu câu Context (length: 51) vào GPU, ta thu về tensor chứa toàn bộ Hidden States từ 36 tầng (+ 1 tầng Embedding ban đầu). Tập trung trực diện vào vị trí Target Token, ta được một ma trận vĩ đại gồm $11100$ hàng ($300 \times 37$) $\times 1280$ chiều nhúng (Dimensions).

### 3.2. Scree Plot và Góc Tối Của Khảo Sát Văn Bản Động (PCA Reality)
Sau khi đưa $11100 \times 1280$ ma trận qua hàm giảm chiều `sklearn PCA()`, Scree Plot phơi bày sự trần trụi của văn xuôi phân tích:
Khác hẳn với việc ép biểu đồ PCA ôm lấy $80\%$ độ tin cậy từ những câu văn "nông cạn" ở phần học trước, hai thành phần chính (Top 2 Principal Components) của bài này chỉ đóng gói được **chưa tới $40\%$** tổng phương sai (Total Variance). 
- Sự giảm mạnh độ bao phủ lý giải rằng dữ liệu văn học cấu trúc thực tại (Real Text) phân tán nội hàm ngữ nghĩa (Semantic Information) phức tạp chéo qua vô vàn các tọa độ ngách. Mất đi $60\%$ nhiễu xung quanh đồng nghĩa bức tranh trực quan 2D sắp tới chỉ là một góc thu hẹp, nó bắt buộc phải được theo dõi kỹ bằng các hàm tương quan thống kê, song song cùng hình ảnh.

---

## 4. Kết Luận Nửa Chặng
Sự khác biệt giữa Token ngôn ngữ và Token mã hóa là trở ngại vĩ đại khi đưa các thư viện cổ điển `spaCy` vào AI Mechanistic. Việc thiết lập hệ thống Filter khắt khe đã dọn đường cho phân tích trọn vẹn không bị nhiễu do chia rẽ Subwords. Với một ma trận PCA chuẩn bị sẵn mang đủ đặc trưng tự nhiên từ văn học, ta sẽ tiếp cận việc tính toán trực tiếp độ dài của Quỹ đạo Không gian ở nấc nghiên cứu trong phần kế tiếp.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm dọn dẹp NLP mismatch và gộp mảng dữ liệu Tensor trong `aero_LLM_08_CodeChallenge Do nouns or adjectives have longer trajectories (part 1).md`.
