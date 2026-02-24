# Thử Thách Lập Trình: Phân Rã Độ Dài Đường Dẫn Luồng Số Dư (Phần 1)

## Tóm tắt (Abstract)
Báo cáo này là bước tổng hòa các khái niệm về Độ dài đường dẫn (Path length) và sự đóng góp của hai mô đun phụ Attention/MLP vào luồng trung tâm (Residual Stream). Sử dụng mô hình GPT-2 Large ($36$ Transformer blocks), thí nghiệm gắn mã theo dõi (Hooks) vào đầu ra chiếu (Projection layers - `c_proj`) của hai khối Attention và MLP để thu thập dữ liệu về lượng "điều chỉnh" (Adjustments) từ mỗi khối trước khi ghi đè trở lại luồng chính. Thực nghiệm đo lường Độ tương đồng Cosine (Cosine Similarity) giữa tập vector điều chỉnh của Attention và MLP chỉ ra rằng: Tại phần lớn các tầng trung gian, hai cơ chế này hoạt động gần như trực giao (Orthogonal) độc lập hoàn toàn. Sự đan chéo đồng tuến chỉ bùng lên ở những chặng mở đầu và kết thúc của kiến trúc.

---

## 1. Mở Đầu (Introduction)
Trong một Transformer Block, vector ban đầu không bị biến đổi thẳng, mà nó chảy dọc theo dòng "Residual Stream". Ở mỗi block, hai nhánh rẽ (Attention và MLP) sẽ "đọc" vector này, giải quyết tính toán cục bộ, và cộng dồn lại vào dòng chính dòng giá trị "đã điều chỉnh" (adjustment vectors). 
* Câu hỏi đặt ra là: Hai cơ chế này có hợp tác hay dẫm chân lên tính toán của nhau không? 
* Nếu Cosine Similarity $\approx 1$: Cả Attention và MLP cùng đẩy Token Embedding về chung một hướng. Nghĩa là chúng tính toán những thông số dư thừa y hệt nhau.
* Nếu Cosine Similarity $\approx 0$: Chúng đóng góp vào luồng chính những mảng kiến thức vuông góc (trực giao) hoàn toàn tách biệt.

Bài thực hành này lập biểu đồ Cosine Similarity xuyên suốt 36 tầng của GPT-2 Large để kiểm chứng mức độ phân chia công việc (Labor division) của mô hình.

---

## 2. Tiền Xử Lý: Kỹ Thuật Gắn Hook Kích Hoạt (Methodology)

### 2.1. Cài Cắm Hooks Vào Module `c_proj`
Thay vì đọc `output_hidden_states` (là tổng hòa sau khi đã cộng), ta cần đo lường *chính xác* thông lượng mà mô đun xả ra:
- Khối Self-Attention Projection: `attn.c_proj`
- Khối MLP Projection: `mlp.c_proj`
Dữ liệu hứng được không phải Token Embeddings gốc, mà chính là vector "Adjustments" quy mô 1280 dimensions của GPT-2 Large.

### 2.2. Trích Xuất Dữ Liệu
Sử dụng phân đoạn tài liệu về triết gia "Nietzsche" (gồm 342 tokens). Việc quét qua 36 Layers giúp tạo ra một ma trận Tensor ba chiều cho cả Attention Adjustments và MLP Adjustments. Số liệu sau đó được đẩy vào công thức PyTorch: `torch.nn.functional.cosine_similarity()`.

---

## 3. Khảo Sát Đánh Giá Dữ Liệu: Thuyết Trực Giao (Analysis)

### 3.1. Theo Dõi Theo Tuyến Tầng (Layer-wise Cosine Similarity)
Trung bình hóa kết quả độ tương đồng theo tầng (kèm thanh Sai Số Chuẩn - Error Bars):
- **Tầng đầu và tầng cuối (Đoạn 0-5 và 30-35):** Ngóc đầu tăng tạo thành bờ vai "Shoulder". Hai cơ chế này biểu hiện mức độ đồng thuật nhẹ (Cosine $\approx 0.1 \to 0.2$), phụ trợ lẫn nhau để nạp hoặc đóng gói từ vựng.
- **Tầng trung tâm (Đoạn 6-29):** Đường phân bổ lún sát vạch 0. Điều này làm sáng tỏ thuyết Cơ chế độc lập (Orthogonality Doctrine). Ở hệ tầng sâu phân tích ngữ cảnh, Attention (chăm lo việc kéo ngữ cảnh xa xôi) và MLP (hoạt động như kho chứa tri thức cục bộ) gần như không chạm mặt nhau, thực hiện độc lập những nhiệm vụ phân rã vector riêng rẽ của mình.

### 3.2. Biểu Đồ Tần Suất (Histogram Distribution)
Trải phẳng toàn bộ ma trận (Flattening $342 \text{ tokens} \times 36 \text{ layers}$), đỉnh tháp Histogram đóng đinh hoàn hảo quanh mốc Cosine $= 0$. Một phần đuôi lệch ngắn (right-skewed tail) nghiêng về phía số dương giải thích cho những lần đồng thuận ở đầu/cuối mô hình. Nhìn chung, kết cấu Transformer không có chỗ cho sự lặp lại thừa thãi.

---

## 4. Kết Luận (Bàn Luận Nửa Chặng)
Hooks là công cụ giải phẫu sắc bén giúp chúng ta mổ xẻ Residual Stream. Bằng cách can thiệp vào tầng `c_proj`, nghiên cứu chứng thực bản thiết kế chia để trị (Divide and Conquer) tối giản nhưng phi thường của Transformer. Việc chứng minh 2 luồng công việc Attention và MLP thẳng góc nhau tạo đòn bẩy vững chắc để mở khóa phần phân tích Độ dài đường dẫn sẽ được diễn giải ở nửa sau của thử thách này.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm cài cắm mã đo lường tại `aero_LLM_04_CodeChallenge Residual stream path length decomposition (part 1).md` (Thiết lập hàm Hook PyTorch cho `c_proj` của khối Transformer ứng dụng GPT-2 Large).
