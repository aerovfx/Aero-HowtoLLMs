# Thử Thách Lập Trình: Biểu Diễn Phân Bố Nhiệt Laminar Của Trọng Số Chú Ý

## Tóm tắt (Abstract)
Thử thách này mở rộng việc khảo sát Cơ chế Chú ý của mô hình từ một Layer đơn lẻ ra toàn bộ kiến trúc đa tầng (Laminar Profile). Bằng cách khai thác mô hình Pythia 2.8 Tỷ tham số (32 Layers) và ứng dụng kỹ thuật Mô hình hóa Mật độ Hạt nhân (KDE), thí nghiệm vẽ ra Bản đồ Phân phối Nóng (Heat maps) biểu diễn hành vi của trọng số Softmax. Biểu đồ xác nhận đặc tính siêu thưa (Sparsity) của trọng số đầu ra và làm rõ sự phân chia trách nhiệm: Các lớp tầng nông có xu hướng phân bổ xác suất đều và thấp để triệt tiêu nhiễu, trong khi các lớp tầng sâu có xu hướng vinh danh cục bộ một vài thẻ Token quá khứ quan trọng để dồn lực dự đoán thẻ từ tiếp theo.

---

## 1. Mở Đầu (Introduction)
Với sự ra đời của hàm Kernel Density Estimation (KDE) ở bài thực hành trước, ta đã chuyển đổi từ biểu diễn rời rạc (Scatter scatter) sang đường cong mật độ mượt (Smoothed PDF). Bước tiến logic tiếp theo là áp đặt phương pháp này xuyên qua dọc trục thời gian của toàn bộ Transformer blocks (Laminar Profile).
Thiết lập thí nghiệm được nâng cấp lên bản thể LLM lớn hơn: Mô hình Pythia 2.8 Tỷ tham số của Eleuther AI. Điều này đòi hỏi người nghiên cứu phải làm quen với sự thay đổi của cấu trúc biến và kỹ thuật cắt nhỏ Ma trận (Splitting Tensor) trong môi trường khối lượng lớn, chuẩn bị cho việc đồ thị hóa hơn 64 hàm mật độ.

---

## 2. Tiết Thiết Lập (Methodology)

### 2.1. Đồng Dạng Cấu Trúc Khác Biệt
Việc chuyển từ kiến trúc OpenAI (GPT-2) sang EleutherAI (Pythia) yêu cầu sự cẩn trọng về biến số. Pythia $2.8B$ được cấu thành từ 32 Layers. Chiều nhúng (Embedding dimensions) $D = 2560$. Tổng số Attention Heads là $N=32$. Kéo theo mỗi Head có chiều không gian $D_{head} = 80$. 
Trong hook tensor thu được, kích thước sẽ bị bóp nghẹt ở quy mô $L \times 7680$. (Bởi vì $2560 \times 3 = 7680$, chứa trọn ổ Query, Key, Value gộp chung).

### 2.2. Vòng Lặp Giải Phẫu Xuyên Lớp (Deep Laminar Extraction)
Kịch bản chạy vòng lặp xuyên 32 tầng Layer. Đối với mỗi Layer, ta làm thao tác `torch.split` để tách Q, K, V. Bước tách tiếp theo dọc theo `dimension=1` trả về $32$ mẻ Head Matrix (Kích thước $SequenceLength \times 80$).
Ta thực hiện nhân Tích vô hướng $QK^T$, chuẩn hóa bằng hệ số $\sqrt{80}$, gắn mặt nạ Causal Masking, và chạy qua hàm kích hoạt $Softmax$.
Kết quả của chuỗi toán học này là Trọng số xác suất (Attention Weights) nằm gọn trong dải $[0, 1]$. Cuối cùng, hàm `scipy.stats.gaussian_kde` nội suy một lưới 300 điểm trên dải $[0, 1]$ để chốt biểu đồ phân phối xác suất.

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Nghịch Lý Nhiễu Loạn Đồ Thị Đường
Khi cố gắng biểu diễn quá trình vận động bằng Đồ thị cắt lớp (Line plots), ta nhận về một ma trận rối mắt với 64 đường cong đè chồng chéo lên nhau (32 đường cho quy luật Tự Chú ý - Self-Attention, và 32 đường cho quy luật Final Token ánh xạ về Previous Tokens). Khả năng phân tích hình thái tầng (Laminar changes) là bất khả thi.

### 3.2. Hiệu Ứng Phân Bố Nhiệt (Heatmap Profiles)
Biện pháp giải quyết là dập phẳng đồ thị thành Ma trận Nhiệt (Heatmaps): Trục X đại diện cho các Tầng Layer ($0 \to 32$); Trục Y đại diện cho giá trị Trọng số Xác suất Softmax ($0 \to 1$); và Màu sắc cường độ sáng (Color Brightness) thể hiện mật độ KDE.
Kết quả đọc được trên Bản đồ nhiệt bộc lộ kiến trúc nhận thức lõi:
- **Tại các khối đầu và giữa (Early/Middle Layers):** Màu sáng trắng (Mật độ siêu dày) quần tụ sát đường Zero, minh chứng cho việc mô hình đang thực hiện ép nén để loại bỏ hoặc cấm đoán Token rác phát nhiễu lộn xộn.
- **Tại các khối sâu (Late Layers):** Mật độ lan rải và một số vệ tinh tỏa sáng nảy lên ở các mốc xác suất cực cao (Ví dụ $0.6 - 0.8$). Ở các trạm biến áp cuối cùng này, mô hình đã dồn toàn bộ nguồn lực "Khám phá" để thâu tóm vào một cụm nhỏ các từ khóa quá khứ sống còn (Relevant context tokens), và phóng to chúng nhằm ra quyết định dự báo Final Output.

---

## 4. Kết Luận
Việc khảo sát Profiler nhiệt độ Attention là minh chứng số học cho triết lý mô hình hóa: Xóa tan nhiễu ở mặt nông và cường thực tín hiệu cốt lõi ở mặt sâu. Tự vận hành Cơ chế Attention từ đầu đến cuối thông qua thuật toán tự nhân ma trận tensor mang tính thực chiến cao. Khi đã nắm vững được biểu đồ mật độ Laminar Profile, ta đã hoàn tất công tác tiền trạm để có thể đi sâu hơn vào việc khai quật những Cụm Circuit thực thi luận lý đặc biệt ở những bài học sau.

---

## Tài Liệu Tham Khảo (Citations)
1. Thử thách đồng nhất Tensor biến thiên trên mô hình Pythia 2.8B trong `aero_LLM_03_CodeChallenge Laminar profile of attention head weights.md`. Mô phỏng cấu trúc trích xuất vòng lặp kép và kỹ thuật đồ họa KDE 2D Heatmaps.
