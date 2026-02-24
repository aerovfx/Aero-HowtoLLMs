# Mối tương quan giữa Hooks và Hidden States: Giải cấu trúc Khối Transformer (Reconstructing Transformer Blocks)

## Tóm tắt (Abstract)
Báo cáo này làm sáng tỏ mối liên hệ cơ học giữa hai phương pháp trích xuất dữ liệu: `output.hidden_states` (quan sát dòng dư - residual stream) và Hooks (quan sát các điều chế nội bộ). Thông qua thực nghiệm tái cấu trúc hoạt hóa của Tầng 11 từ Tầng 10 trên mô hình GPT-2, nghiên cứu chứng minh rằng đầu ra của một khối Transformer chính bằng tổng của đầu vào và các "độ lệch" (deltas) được tính toán bởi phân đoạn Attention và MLP. Báo cáo cũng nhấn mạnh tầm quan trọng của việc hiểu các chế độ `eval` so với `train` trong PyTorch khi làm việc với Hooks, đặc biệt là vai trò của hàm `detach()`.

---

## 1. Mở Đầu (Introduction)
Một trong những nguyên lý cốt lõi của kiến trúc Transformer là mạng lưới các kết nối dư (residual connections). Thay vì biến đổi hoàn toàn vector nhúng ở mỗi tầng, mô hình chỉ tính toán các "điều chỉnh" (adjustments) nhỏ dựa trên ngữ cảnh và tri thức thế giới, sau đó cộng dồn chúng vào dòng chảy thông tin. Báo cáo này sẽ thực chứng nguyên lý đó bằng cách kết hợp dữ liệu từ Hidden States và Hooks.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Vị trí Cấy Hook (C_proj)
Để lấy được "độ lệch" cuối cùng của mỗi phân đoạn, chúng ta cấy Hook vào lớp `c_proj` (output projection) của cả Attention và MLP. Đây là điểm cuối cùng trước khi các giá trị điều chỉnh được cộng ngược trở lại vào residual stream.

### 2.2. Quản lý Đồ thị Tính toán (Gradient Detachment)
Khi mô hình không ở chế độ `eval()`, các tensor trích xuất qua Hook vẫn mang theo thông tin về gradient và đồ thị tính toán. 
- **Kỹ thuật:** Sử dụng `.detach()` để tách các con số thuần túy khỏi đồ thị, giúp tiết kiệm bộ nhớ và tránh các lỗi tính toán không mong muốn trong quá trình phân tích hậu kỳ (post-processing).
- **Lưu ý:** Nếu sử dụng `eval()`, tính năng tính gradient bị tắt hoàn toàn, giúp lược bỏ bước detach này.

---

## 3. Kết Quả Thực Nghiệm: Tái cấu trúc Hoạt hóa

### 3.1. Sự bảo tồn Tín hiệu (Laminar Correlation)
Đồ thị phân tán giữa đầu ra của Tầng 10 và Tầng 11 cho thấy sự tương quan cực mạnh ($r \approx 1.0$). Điều này khẳng định rằng Hidden State không bị thay đổi hoàn toàn sau mỗi Transformer Block mà chỉ bị biến đổi nhẹ.

### 3.2. Công thức Tái cấu trúc (The Reconstruction Formula)
Giá trị hoạt hóa của Tầng $L+1$ có thể được dự đoán chính xác tuyệt đối bằng công thức:
$$ \mathbf{H}_{L+1} = \mathbf{H}_L + \Delta_{Attention} + \Delta_{MLP} $$
Thực nghiệm cho thấy khi cộng các giá trị trích xuất từ Hook ($\Delta$) vào Hidden State hiện tại, ta thu được kết quả khớp hoàn hảo với Hidden State của tầng tiếp theo trích xuất từ `output.hidden_states`.

### 3.3. Hiện tượng Ngoại lai (Outlier Handling)
Quan sát thực nghiệm cho thấy một số chiều (thường ở token đầu tiên) có giá trị kích hoạt cực lớn. Trong phân tích, việc sử dụng các mặt nạ (masks) để loại bỏ các giá trị ngoại lai này giúp việc quan sát sự tương quan của 99% dữ liệu còn lại trở nên rõ ràng hơn trên đồ thị.

---

## 4. Thảo Luận: Vai trò của Dropout trong Phân tích
Nếu giữ mô hình ở chế độ `train`, các lớp Dropout vẫn hoạt động, gây ra sự ngẫu nhiên trong các kích hoạt trích xuất. Điều này nhắc nhở các nhà nghiên cứu rằng trạng thái của mô hình (Mode) có ảnh hưởng quyết định đến tính lặp lại của các phép đo cơ học.

---

## 5. Kết Luận
Sự khớp nối hoàn hảo giữa Hooks và Hidden States khẳng định tính đúng đắn của mô thức "dòng chảy dư" (residual stream hypothesis). Việc hiểu rõ cách các thành phần cộng dồn vào nhau cung cấp công cụ để thực hiện các can thiệp sâu hơn, như việc cô lập và chỉnh sửa một thành phần đơn lẻ (ví dụ: chỉ chỉnh sửa MLP delta) để quan sát tác động lan tỏa hạ nguồn.

---

## Tài liệu tham khảo (Citations)
1. Giải thuật tái cấu trúc khối Transformer bằng Hooks dựa trên `aero_LLM_06_Relation between hooks and output.hidden_states.md`. Phân tích sự đóng góp của Attention và MLP deltas vào residual stream.
