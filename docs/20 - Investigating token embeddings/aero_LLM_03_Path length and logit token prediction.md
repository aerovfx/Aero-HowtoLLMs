# Đo Lường Độ Dài Đường Dẫn (Path Length) Sự Tương Quan Với Dự Đoán Token

## Tóm tắt (Abstract)
Bên cạnh việc đo lường Góc dịch chuyển (Angle Rotations), độ lớn của sự dịch chuyển không gian vector có thể được định lượng hoá bằng Độ dài đường dẫn (Path Length) – chỉ số Euclidean distance đo trực tiếp khoảng cách mà vector nhận thêm khi qua khỏi một Block Transformer. Nghiên cứu thực nghiệm trên 4 phiên bản kiến trúc GPT-2 (Small, Medium, Large, XL) và đánh giá độ chênh lệch Path Length tại trạm trung chuyển cuối cùng (Tầng Penultimate $11 \to 12$). Đáng chú ý, ở cấu hình GPT-2 Small, mức độ tĩnh lặng (Path length ngắn) ở tầng cuối tỉ lệ nghịch với xác suất sinh từ chính xác (Logits for Next Token Prediction). Tuy nhiên, hiện tượng trượt dốc tuyến tính này từ chối lặp lại đồng nhất trên các mô hình siêu tham số lớn hơn (Medium/Large/XL), mở ra câu hỏi lớn về tính tương đối trong việc suy diễn cơ học LLM.

---

## 1. Mở Đầu (Introduction)
Các nỗ lực ở các phần trước tập trung lý giải làm cách nào một mạng Transformer bẻ hướng (rotation) vector ẩn. Tuy nhiên, góc biểu diễn không bảo tồn thông tin cường độ co giãn (magnitude scaling) của vector (Ví dụ: một điểm dậm chân nguyên phương nhưng kéo dài gấp đôi sẽ không lưu lại bất cứ biến thiên Góc nào). Do đó, chỉ số thứ hai được đề ra là Độ dài Đường Dẫn (Path Length) - một đại lượng tích hợp cả thông số xoay chiều lẫn chênh lệch độ dài. 
Giả thuyết đặt ra: Ở các trạm biến áp cuối cùng trước khi ra khỏi mạng lưới phân giải, nếu mô hình đã tích luỹ đủ chắc chắn về từ ngữ sắp dự đoán, nó sẽ không "ngọ nguậy" mạnh nữa, tức vector sẽ gần như đứng yên (Path length siêu nhỏ).

---

## 2. Nền Tảng Toán Hình Học & Logic Đo Lường (Methodology)

### 2.1. Định Tuyến Khoảng Cách (Euclidean Path Length)
Giả định Vector trích xuất của token ở mốc Layer $L_i$ là $x$ và ở Layer $L_{i+1}$ kế tiếp là $y$. Khoảng cách tịnh tiến Path Length là hiệu số độ dài Norm của chúng:
$$ \text{Path Length} = \|y - x\| = \sqrt{\sum (y_i - x_i)^2} $$
Trong code diễn dịch, tham số này được khởi chạy qua phép trừ trực tiếp Tensor và tính toán chuẩn Normalize 2 (`torch.norm`).

### 2.2. Dữ Liệu Input (Targeted Setup)
Sử dụng đoạn tóm tắt từ Wikipedia về triết gia "Nietzsche", làm trạm thử nghiệm. Quá trình tính toán diễn ra ở từng mốc Hidden States. Kèm theo đó, thuật toán truy vấn hàm Lũy kề (Cumulative Path Length: `np.cumsum()`) để theo dõi tốc độ trương nở quãng đường qua từng lớp. Ngoài ra số liệu Path Length từ block áp chót (Penultimate lên Ultimate Block) được mang ra so sánh (Pearson/Spearman Correlation) trực diện với tham số Logits chốt hạ (Next token prediction logics).

---

## 3. Khám Phá Trực Quan (Analysis & Visualizations)

### 3.1. Sự Tăng Trưởng Lũy Kế Của Đường Dẫn (Cumulative Path Length)
Bản đồ phân tán lũy kế hiển thị một sự gia tăng tuyến tính đồng điệu. Nhìn chung, Vector của mô hình không ngừng bị kéo dài và kéo dạt đi xa mãi qua các Transformer blocks, không hề có hiện tượng bị dồn cục hội tụ tại một "center". Năng lượng liên tục được bơm vào (nhờ ResNet / Add & Norm logic).

### 3.2. Điểm Mù Của Sự Sao Chép Tương Quan (The Replication Nuance)
Mô hình GPT-2 Small ném ra một kết quả tương quan nghịch (Negative Correlation) tuyệt đẹp: Path Length ở chặng $11 \to 12$ càng xấp xỉ 0 thì Logits của chữ được phát ngôn ra càng mang trọng lượng lớn. Nghĩa vụ lý luận rất rõ: Nếu mô hình không cần phải xê dịch nhiều ở khúc chót, nó cực kỳ tự tin vào đáp án của mình.
Mặc dù vậy, hệ quy chiếu này gãy vỡ phũ phàng khi quét qua 3 bản thể cao cấp hơn (Medium, Large, XL):
- Biểu đồ phân tán (Scatter Plot) nát ra thành các cụm nhiễu, Correlation bốc hơi ($r \approx 0$).
- Dù vậy, nó không chối bỏ hoàn toàn sự hữu ích của giả thuyết Path Length. Ở những ma trận đa tầng với sức chứa lớn như XL, "sự tự tin" không chỉ thể hiện qua một vector phẳng, mà có thể do các cơ chế song song (Parallel Head Operations) đong đếm và khỏa lấp nhiễu lệch đi nhau. 

---

## 4. Kết Luận
Path Length là một lát cắt bổ sung hoàn hảo với Angular Rotation để bắt giữ những chuyển động tĩnh/bóp nén/bành trướng của Vector sau khi bị Transformer vặn xoắn. Thực nghiệm Correlation trên mô hình Small với hiện tượng thiếu nhất quán trên mô hình XL không mang nghĩa phủ định phương trình, ngược lại, nó là lời nhắc nhở sinh động nhất về Giới hạn Ngoại suy (Extrapolation Limitations) trong việc "giải phẫu cơ học" diễn dịch: Mô hình khác quy mô sẽ hình thành những luồng tư duy và phong cách phân bổ ma trận tín hiệu không hề giống nhau.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm đo đạc khoảng cách Norm và tính toán Lũy kề tại tập code `aero_LLM_03_Path length and logit token prediction.md` (Quét song song vòng lặp 4 mô hình GPT-2 Small tớt XL).
