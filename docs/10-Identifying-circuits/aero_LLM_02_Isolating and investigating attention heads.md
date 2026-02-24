# Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)

## Tóm tắt (Abstract)
Báo cáo trình bày phương pháp giải phẫu một trong những linh hồn cốt lõi của kiến trúc Transformer: Cơ chế Đa chú ý (Multi-head Attention). Bằng việc theo dõi cách biến đổi và phân mảnh ma trận Tương tác $Q, K, V$ dọc theo hệ chiều nhúng thành các module đầu độc lập (Heads), ta có thể nhận thức được sự chuyên biệt hóa luồng thông tin của LLM. Một phát hiện đáng chú ý là sự thiên vị điểm âm (Negative shift) của các Tích vô hướng gốc (Raw Attention Scores), điều này giải thích toán học cho cơ chế "Sparsity" - triệt tiêu sự nhiễu loạn từ token không liên quan. Báo cáo cũng đề xuất phương pháp mô hình hóa mật độ hạt nhân Kernel Density Estimation (KDE) thay cho Scatter plots nội suy, giúp trực quan hóa phân bổ xác suất một cách khoa học.

---

## 1. Mở Đầu (Introduction)
Trong Mạng Mạch của Deep Learning, việc quy tụ hàng khối Head lại với nhau thông qua ma trận trộn tuyến tính $W_o$ (Linear mix matrix) là chìa khóa tổng hợp kiến thức ngôn ngữ. Tuy nhiên, nếu chúng ta có thể chẻ nhỏ và truy cập vào từng "Não bộ phụ" (Head) riêng lẻ đang phân tích gì, ta sẽ hiểu được cơ chế hoạt động vi mô (Mechanistic Interpretability). Công việc này đòi hỏi kết hợp phương trình Attention cốt lõi: $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ kết hợp thao tác ma trận tinh vi.

---

## 2. Tiết Thiết Lập (Methodology)

### 2.1. Nhắc Lại Thuật Toán Attention và Mặt Nạ Causal Label (Masking)
$Q$ (Query) đại diện cho "Token hiện tại đang tìm kiếm gì?", còn $K$ (Keys) đại diện cho "Các token cũ giữ thông tin gì đáng giá?". Tích vô hướng $QK^T$ đo lường sự tương thích. 
Tuy nhiên, Transformer là bộ dự báo chuỗi theo thời gian (Autoregressive), nó bắt buộc không được "Nhìn trộm" tương lai. Lớp Mặt Nạ $M$ (Masking matrix) được phủ lên $QK^T$: Các tọa độ ở tam giác dưới (Quá khứ) nhận mức $1$, tọa độ ở tam giác trên (Tương lai) nhận $-\infty$. Khi qua hàm phi tuyến kích hoạt $Softmax$, hàm số $e^{-\infty}$ biến mất thành điểm $0$ tuyệt đối. 
*Hệ quả dị biệt:* Mảnh Token đầu tiên của chuỗi không có quá khứ, nên toàn bộ thông số liên kết ngược bị xóa sổ $\to$ tự gán $100\%$ lực chú ý vào chính bản thân nó (Outlier error).

### 2.2. Trích Xuất Attention Đầu Phụ (Heads Isolation)
Trên GPT-2 Small, ma trận sau khi hook lấy cắp từ `hook_h.attn.c_attn` sẽ là một khối $768 \times 2304$. Do $2304 = 768 \times 3$, nó đang gộp chung tệp $Q, K, V$. 
1. Cắt lấy $1/3$ đầu tiên ta được Ma trận thuần Query $Q$ (Kích thước: $SequenceLength \times 768$).
2. Tiếp tục dùng hàm `torch.split` chiếu dọc theo chiều Dimensions (768), văm thành $12$ khúc. Kết quả: $12$ Attention Heads, mỗi Head thu được ma trận $SequenceLength \times 64$.
3. Tại điểm này, ta tính Tích vô hướng (Dot products) nội bộ cho từng Head riêng biệt để lấy Raw Attention Scores.

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Sự Thiên Vị Âm Tính Vô Hướng (Negative Raw Attention Shift)
Theo lý thuyết xác suất, khi lấy mẫu ma trận điểm nhân với nhau, phân bổ đồ thị phân tán (Scatter plots) của $QK^T$ (Raw attention scores) thường nên nằm ở dạng đối xứng ngay quanh mốc zero. Tuy nhiên, GPT-2 điều hướng trọng số lệch mạnh mẽ về khu vực cực âm (Negative numbers). 
Đây không phải là lỗi. Nó là một thủ thuật Tối ưu Thưa (Sparsity mechanism). Khi số gốc mang giá trị âm sâu, hàm kích hoạt $Softmax$ sẽ dập toàn bộ tập xác suất này xấp xỉ mức $0$. Việc LLM đẩy hầu hết điểm tương tác xuống mức âm giúp triệt tiêu hoàn toàn các mối quan hệ Token dư thừa từ quá khứ (suppression), qua đó để nhường chỗ, vinh danh cho một số rất nhỏ các kết nối ngữ pháp thực sự ý nghĩa (Ví dụ: tính từ liên kết danh từ).

### 3.2. Nội Suy Phân Bổ Mật Độ KDE (Kernel Density Estimation)
Phương thức biểu diễn bằng các chấm phân tán Scatter plots trở nên vô dụng nếu dữ liệu lớn cồng kềnh qua hàng chục Layers. Phương thức thay thế: KDE (Mô hình hóa mật độ hạt nhân).
KDE coi một điểm phân tán là một tâm thu hút phân phối vi mô (Gaussian blur). Bằng cách convolve lặp và cộng dồn toàn bộ các màng sương Gaussian có độ băng thông nhất định (Bandwidth parameter), ta biến các số thô (Discrete values) thành đường cong phổ phân bổ mượt mà (Probability distribution curve). 

---

## 4. Kết Luận
Việc tách lẻ các Head giải phẫu quá trình tính tương phản Query-Key đưa lại lời giải đáp vì sao $Softmax$ có năng lực xử lý ngôn ngữ sạch sẽ và sắc bén: Nhờ mô hình tự động "Dìm" phổ Tích vô hướng gốc về các chỉ số siêu nhỏ để loại bỏ nhiễu. Phương pháp tách chẻ ma trận trực tiếp và áp dụng hệ tính toán mật độ hạt nhân (KDE) là bậc thang dữ liệu hoàn hảo trước khi đi sâu vẽ dải viền (Laminar Profiles) Attention head, bước cơ bản để khám phá "Mạng mạch" ở mô-đun kế tiếp bài thử thách.

---

## Tài Liệu Tham Khảo (Citations)
1. Cơ chế cắt mảnh ma trận và phân chia Tensor trong `aero_LLM_02_Isolating and investigating attention heads.md`. Thí nghiệm vẽ KDE thông qua thư viện `scipy.stats.gaussian_kde` và minh họa dịch chuyển âm Tích vô hướng.
