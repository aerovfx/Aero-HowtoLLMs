# Thử Thách Lập Trình (Code Challenge): MI Và Khoảng Cách Token (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này giải quyết bài toán định lượng thông tin tương hỗ giữa các "ngữ cảnh nội bộ" (local context) của một nhóm các cụm từ giống hệt nhau, chạy trên mô hình khổng lồ GPT-2 XL (48 Layers). Thông qua đoạn văn bản mồi thuộc chủ đề Cà phê Thổ Nhĩ Kỳ, có tổng cộng 7 lần từ "coffee" lặp lại. Thay vì đo lường Mutual Information (MI) dọc trên các Token, ta đo lường MI giữa 7 từ "coffee" này kết nối qua $1600$ chiều ẩn (Hidden Dimensions). Đồng thời, bài thực nghiệm giới thiệu kỹ thuật loại bỏ biệt lệ (Outliers Trimming) bằng Z-Score và sử dụng Hệ số tương quan hạng Kendall (Kendall's Tau) để khám phá mối liên hệ nghịch biến giữa Độ lớn của MI và Khoảng cách vị trí của hệ Token.

---

## 1. Mở Đầu (Introduction)
Trong việc xử lý ngôn ngữ tự nhiên, một từ đơn lẻ (Ví dụ "coffee") có thể lặp lại nhiều lần trong đoạn văn, mỗi lần lại mang một tiểu ngữ cảnh (local context) hơi khác nhau. Điều này đặt ra câu hỏi hấp dẫn:
*"Với cùng một gốc Token id, biểu diễn kích hoạt $Attention$ ở các vị trí khác nhau có chia sẻ thông tin gì không? Và nếu chúng xa nhau về mặt vật lý, liệu khả năng mang tin cấu trúc có bị sụt giảm không?"*

Để tìm lời giải, chúng tôi khai thác mạng GPT-2 XL, tập trung vào kết xuất cuối của Attention Block (được gọi là $c\_proj$).

---

## 2. Tiền Xử Lý Dữ Liệu Và Quy Trình Loại Bỏ Nhiễu (Outliers)

### 2.1. Nạp Hàm Kích Hoạt "Target Words"
Mô hình nhập nội dung đoạn văn có chứa 7 lần xuất hiện từ "coffee". 
Tại Layer 3, dữ liệu kích hoạt của mỗi từ "coffee" tương ứng là một vector dài 1600 chiều (1600 Dimensions). Chúng ta tiến hành vẽ Scatter Plot so khớp Vector thứ $1$ và Vector thứ $3$.

### 2.2. Xử Lý Các Điểm Dữ Liệu Cực Đoan (Extreme Values)
Khi quan sát biểu đồ hoạt động của mạng LLM, thường xuất hiện khoảng 1-2 điểm nhiễu (neurons) có cường độ kích hoạt "phóng vút" lên rất cao so với đám mây phân bổ trung tâm. Mặc dù đây là các tín hiệu mạng bình thường (không phải lỗi bộ nhớ), hiện tượng cực đỉnh (extreme values) lại phá nát các thuật toán đo chia Histogram của MI.

**Cách khắc phục:** Không gian hóa Z-Score. 
$$ Z = \frac{x_i - \bar{x}}{\sigma} $$
Áp dụng Z-score cho cả 2 vector. Bất kỳ giá trị nào có $|Z| > 4$ (Vượt quá 4 lần độ lệch chuẩn) sẽ bị gán cờ Outlier và dạt bỏ khỏi danh sách đo MI. 
Việc cắt tỉa dữ liệu thừa (Trimmed Data) này giúp đẩy MI từ một con số bị dìm do nhiễu $\to$ phục hồi lại điểm tương hỗ cốt lõi, phản biện lại nhược điểm của công thức histogram Manual.

---

## 3. Khoảng Cách Vị Trí Vs Tương Quan Thông Tin (Analysis & Results)

### 3.1. Tính Ma Trận Tương Hỗ Chéo Điểm (Pairwise Token MI Matrix)
Vì có 7 mục tiêu, ma trận phân tích sẽ có cấu trúc $7 \times 7$. Bỏ qua chéo chính và nửa dưới đối xứng, phần dữ liệu nửa trên chứa $MI$ giữa toàn bộ các cặp khoảng cách từ 1 đến 7. 

### 3.2. Ma Trận Khoảng Cách Cục Bộ (Inter-token Distances)
Khoảng cách vật lý giữa hai từ "coffee" được tính giản lược bằng số lượng Token nằm xen giữa chúng. Không phải Embedding Vector Distances. (Do đây là số nguyên bậc thứ tự, không phải biến thiên liên tục).

### 3.3. Phương Trình Tương Quan Xếp Hạng Kendall (Kendall's tau)
Vì biến quãng cách là một chuỗi mang tính định hạng (ordinal variable - số nguyên ngắt quãng), việc dùng Tương quan Pearson là sai nguyên lý thống kê. Ta phải chuyển qua hệ số **Kendall's Tau** (Tương tự Pearson, chạy từ $-1 \to 1$).

**Kết quả Scatter Plot kết nối:**
Biểu đồ trải hiển thị mối tương quan nghịch đảo rõ rệt $\to$ `Hệ số r Kendall = -0.5`. 
- **Giải thích:** Hai từ "coffee" đứng càng gần nhau trong một câu, chỉ số M.I giữa biểu diễn không gian $Attention$ của chúng càng mãnh liệt. Khi hai từ bị đẩy ra xa nhau chừng vài chục định vị, tiểu lớp ngữ cảnh bị vỡ vụn, khiến khả năng san sẻ tương đồng ý niệm rơi thẳng đứng.

---

## 4. Kết Luận
Bài toán Token Distance vén màn cơ chế "Nhớ gần" (Local Memory Context) của Multi-head Attention thông qua thấu kính Mutual Information. Bằng việc chắt lọc Z-score Outliers, ta có thể xây dựng các biểu đồ tương tự Pearson nhưng dành cho các đại lượng phi tuyến cực kỳ chính xác. Ở phần sau, nghiên cứu sẽ phát triển mô hình này mở rộng xuyên suốt 48 Blocks (Laminar Profile) để xem xét định kiến nội dung ở vùng biến đổi sâu nhất (Deep Layers).

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu lệnh code trích xuất từ thí nghiệm: `aero_LLM_11_CodeChallenge Attention to coffee MI and token distances (part 1).md` (Giới thiệu hàm tính Z-Score $>4$, Kendall tau Correlation và nguyên lý MI của Token cặp).
