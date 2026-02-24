# Phân Khảo Cấu Trúc Cụm (Clusters): Dấu Câu Nội Bộ vs Dấu Câu Kết Thúc Tập 1

## Tóm tắt (Abstract)
Báo cáo này tập hợp các phép đo sự phụ thuộc thống kê (Statistical Dependencies) – bao gồm Mutual Information (MI) và Covariance – áp dụng trên biểu diễn nhúng (embeddings) của các dấu câu. Bằng phương pháp chia tách rành mạch Dấu câu nội bộ (Internal Punctuation - như dấu phẩy) và Dấu câu kết thúc (Terminal Punctuation - như dấu chấm), mồi văn bản từ tiểu thuyết "Heart of Darkness" chạy qua mạng GPT-2 Medium chỉ ra rằng Covariance cực kỳ nhạy bén trong việc bắt lỗi các cụm (clusters) dữ liệu nội bộ ở những tầng đầu, trong khi M.I hướng đến một dải tuyến tính đồng đều hơn. Nghiên cứu cũng bàn luận về Nghịch lý Simpson trong đo lường tương quan hỗn hợp.

---

## 1. Mở Đầu (Introduction)
Dấu câu đóng vai trò như những "trạm kiểm soát luồng" trong lập trình ngôn ngữ tự nhiên:
- **Dấu câu nội bộ (Internal Punctuation):** như dấu phẩy (,), dấu chấm phẩy (;). Chúng rẽ nhánh cục bộ nhưng giữ nguyên đại ngữ cảnh của câu.
- **Dấu câu kết thúc (Terminal Punctuation):** như dấu chấm (.), dấu chấm hỏi (?), dấu chấm than (!). Chúng đóng gói toàn bộ luồng thông tin hiện tại và dọn sạch bộ đệm (context window) để chào đón một tư tưởng mới.
Nghiên cứu này kỳ vọng trả lời câu hỏi: *Với hàng chục ngàn chiều kích hoạt (activations), LLM xử lý sự khác biệt cấu trúc phân tách này như thế nào nếu ta dùng hai lăng kính toán học Mutual Information và Covariance để nội soi?*

---

## 2. Tiền Xử Lý Dữ Liệu (Methodology)

### 2.1. Phân Loại Dấu Câu (Punctuation Extraction)
Tác phẩm "Heart of Darkness" (với hơn 64.000 tokens) được dùng làm kho ngữ liệu. Thông qua mã lệnh dò tìm, ta thiết lập 3 hệ cờ phân loại (Flags):
- 0: Không phải dấu câu.
- 1: Dấu câu nội bộ (Chủ chốt là Comma). (Lọc ra $\approx$ 3000 mẫu).
- 2: Dấu câu kết thúc (Chủ chốt là Period). Để tránh nhầm lẫn với dấu thập phân của chữ số, các thuật toán kiểm tra chuỗi token liền kề được áp dụng. (Lọc ra $\approx$ 2000 mẫu).

### 2.2. Kích Hoạt Tensor Và Điều Chỉnh Context Window
Thay vì nạp toàn bộ câu, các trạm Batch được chẻ nhỏ theo công thức: Lấy 20 tokens đằng trước dấu câu (Pre-context) và 10 tokens đằng sau (Post-context), tổng 31 tokens. GPT-2 Medium đẩy 250 đoạn trích ngẫu nhiên cho **Internal** và 250 đoạn trích cho **Terminal** quy nạp vào GPU, sau đó vector Hidden States (`250 x 31 x 1024`) được đưa trở lại CPU và chuyển đổi thành Numpy Matrix để mổ xẻ. 

---

## 3. Khảo Sát Tương Quan (Analysis & Results)

### 3.1. Sự Phân Cụm Của Covariance Ở Các Tầng Nông (Shallow Layers)
Tiến hành tính ma trận đồng biến (Pairs Matrix) cho lớp đầu tiên (Layer 1 - ngay sau tầng nhúng Embeddings):
- **Với Mutual Information:** Quần thể các điểm giao thoa Mutual Information tạo thành một khối nhiễu phân bổ (Blob) duy nhất.
- **Với Covariance:** Đồ hình scatter plot vỡ ra thành ba cụm (Clusters) rõ rệt đối với mảng *Internal Punctuation*. 
Hiện tượng này phát tín hiệu: Ở những bước đầu tiếp xúc với ngôn ngữ, LLM tách định nghĩa "dấu câu nối" thành 3 biểu diễn kích hoạt điện toán hoàn toàn tách biệt, trong khi "dấu kết câu" được co cụm đồng nhất. 

### 3.2. Hiệu Ứng Hội Tụ Ở Tầng Sâu (Deep Layers)
Tuy nhiên, cấu trúc 3 cụm (3-clusters) của Covariance không phải là một "hiện tượng bền vững vĩnh cửu". Khi tịnh tiến thuật toán lên Layer 20, ba vùng đốm phân tán này hoà quyện lại thành 1 cụm thống nhất. Tính năng phân tách chi ly của cụm "dấu câu rẽ nhánh" không còn cần thiết đối với tầng học sâu - nơi mạng Neural ưu tiên gộp mọi chỉ số token lại thành một định dạng xác suất tổng quát nhằm tìm kiếm các Word-Logits kế tiếp. 

### 3.3. Nghịch Lý Simpson (Simpson's Paradox)
Điều kỳ lạ diễn ra khi lồng ghép cả 2 chỉ số lên 1 mặt phẳng (Covariance trục X và M.I trục Y). Ở Layer 1, thông số Tương quan Pearson tổng thể báo hiệu $-0.13$ (Tương quan nghịch yếu). Nhưng đi sâu vào từng cụm con một cách độc lập, đường hồi quy biên độ lại có xu hướng Tương quan đồng biến (Âm sinh Âm, Dương sinh Dương).
Sự nhầm lẫn chỉ số sinh ra do sự phân nhánh ngầm (Subgroups confounder) được gọi là **Nghịch lý Simpson**. 

---

## 4. Kết Luận
Covariance phô diễn lợi thế vượt trội khi phát hiện xu hướng rẽ nhánh nhóm mầm trong tầng đáy mô hình. Còn Mutual Information duy trì độ ổn định đo lường dải thông tin một cách tổng quát. Sự mâu thuẫn giữa quy mô Cụm nhỏ (Subgroups) và Tổng thể (Global Data) nhắc nhở việc dán nhãn tính chất biểu diễn (Representations) LLM phải luôn song hành với hiểu biết về hiện tượng thống kê Simpson.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu mã code thuộc `aero_LLM_13_CodeChallenge Clusters in internal vs. terminal punctuation (part 1).md` (Hướng dẫn quy hoạch context arrays kích thước 31 tokens và xử lý nghịch lý nhóm Simpson).
