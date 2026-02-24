# Thử thách Lập trình: Hình chiếu MLP Điều chỉnh theo Danh mục (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này trình bày giai đoạn đầu của một nghiên cứu chuyên sâu về các đơn vị hình chiếu (projection units) trong lớp MLP của mô hình GPT-2 Large. Mục tiêu là xác định xem các nơ-ron ở lớp co (contraction layer) có bộc lộ tính chọn lọc đối với các danh mục ngữ nghĩa cụ thể hay không. Nghiên cứu thực hiện trên hai lĩnh vực từ vựng đối lập: "Vệ sinh nha khoa" (Dental hygiene) và "Đồ nội thất" (Furniture). Quy trình bao gồm việc giải quyết bài toán trích xuất hoạt hóa từ các từ đa token (multi-token words) và triển khai kiểm định thống kê T-test diện rộng trên 36 tầng của mô hình. Kết quả sơ bộ xác nhận sự tồn tại của các nơ-ron điều chỉnh theo danh mục (category-tuned neurons) với xu hướng giảm dần mật độ khi đi sâu vào các tầng cuối của mạng.

---

## 1. Mở Đầu (Introduction)
Trong các phần trước, chúng ta đã khám phá nơ-ron MLP ở lớp mở rộng (expansion layer). Tuy nhiên, các nơ-ron hình chiếu (lớp `c_proj`) – nơi nén thông tin trở lại kích thước của residual stream – cũng đóng vai trò quan trọng trong việc truyền dẫn các tính năng ngữ nghĩa đã được trích xuất. Thử thách này đặt ra câu hỏi: Liệu mô hình có "dành riêng" các nơ-ron hình chiếu nhất định để mã hóa các khái niệm như "bàn ghế" hay "đồ dùng nha khoa"?

---

## 2. Thiết lập Thực nghiệm (Methodology)

### 2.1. Chuẩn bị Dữ liệu và Danh mục
- **Mô hình:** GPT-2 Large (embedding dimension $d=1280$, 36 transformer blocks).
- **Danh mục ngữ nghĩa:**
    1. *Vệ sinh nha khoa:* toothpaste, toothbrush, floss, mouthwash.
    2. *Đồ nội thất:* doorknob, dishwasher, cupboard, bookshelf.
- **Cấu trúc dữ liệu:** 40 câu văn (20 câu cho mỗi danh mục), mỗi câu chứa đúng một từ đích (target word) ở các vị trí ngẫu nhiên.

### 2.2. Thuật toán Xác định vị trí và Trích xuất
Do hầu hết các từ đích là "multi-token" (ví dụ: "toothpaste" $\rightarrow$ ["tooth", "paste"]), nghiên cứu áp dụng quy tắc "token cuối cùng" đã được chứng minh ở bài trước. Một vòng lặp tịnh tiến (sliding window match) được triển khai để xác định chính xác index của token kết thúc khái niệm trong mỗi câu, đảm bảo vector hoạt hóa thu được đã tích hợp đầy đủ ngữ nghĩa của cụm từ.

### 2.3. Cấu hình Hooks
Hooks được cấy vào thành phần `c_proj` của tất cả 36 khối Transformer. DỮ liệu thu thập là một tensor 3 chiều kích thước $[36, 40, 1280]$ đại diện cho (Layers, Sentences, Neurons).

---

## 3. Phân tích Thống kê (Statistical Analysis)

### 3.1. Kiểm định T-test diện rộng
Sử dụng `scipy.stats.ttest_ind` (kiểm định T độc lập) để so sánh hoạt hóa của từng nơ-ron ($1280 \times 36 = 46.080$ phép thử) giữa nhóm "Nha khoa" ($n=20$) và nhóm "Nội thất" ($n=20$). 

### 3.2. Hiệu chỉnh Đa so sánh
Áp dụng hiệu chỉnh Bonferroni trong phạm vi mỗi tầng: $\alpha_{adj} = 0.05 / 1280$. Mặc dù ngưỡng này là khắt khe, nhưng các đặc tính của phân phối T đảm bảo rằng chúng ta vẫn xác định được các tín hiệu vượt trội so với nhiễu.

---

## 4. Kết Quả Quan Sát

### 4.1. Phân phối Nơ-ron có Ý nghĩa Thống kê
- **Mật độ:** Khoảng 5% đến 10% số nơ-ron trong mỗi tầng bộc lộ tính chọn lọc danh mục rõ rệt.
- **Xu hướng theo Tầng:** Tỷ lệ nơ-ron "nhạy cảm" với danh mục có xu hướng giảm dần ở các tầng cuối. Điều này ủng hộ giả thuyết rằng ở giai đoạn cuối của quá trình xử lý, mô hình ưu tiên việc chuẩn bị logits cho token tiếp theo hơn là duy trì các biểu diễn phân loại của token hiện tại.

### 4.2. Tính Đối xứng của T-values
Kết quả cho thấy sự phân bổ cân bằng giữa các T-value dương (ưu tiên Dental hygiene) và âm (ưu tiên Furniture), chứng tỏ lớp hình chiếu MLP chứa các đơn vị chuyên biệt hóa cho cả hai lĩnh vực ngữ nghĩa một cách độc lập.

---

## 5. Kết Luận Phần 1
Chúng ta đã chứng minh được rằng lớp co của MLP không chỉ là một phép biến đổi tuyến tính đơn thuần mà còn chứa các "kênh" chuyên biệt hóa cho ngữ nghĩa phạm trù. Giai đoạn tiếp theo sẽ tập trung vào việc kiểm chứng tính bền vững (robustness) của các nơ-ron này trên những tập dữ liệu hoàn toàn khác để loại trừ khả năng quá khớp (overfitting).

---

## Tài liệu tham khảo (Citations)
1. Thử thách về Category-tuned MLP projections trên GPT-2 Large dựa trên `aero_LLM_15_CodeChallenge Category-tuned MLP projections (part 1).md`. Thiết lập bài toán multi-token và phân tích T-test xuyên tầng.
