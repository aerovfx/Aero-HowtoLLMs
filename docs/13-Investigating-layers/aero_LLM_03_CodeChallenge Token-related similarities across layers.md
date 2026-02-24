# Thử Thách Lập Trình (Code Challenge): Phân Tích Độ Tương Đồng Của Token Xuyên Suốt Các Tầng Ẩn

## Tóm tắt (Abstract)
Kế thừa và mở rộng từ các kỹ thuật tính toán trong phần 1 và 2, bài viết này trình bày phương pháp mở rộng thực nghiệm để phân tách toàn bộ thông lượng kích hoạt (activations) xuyên suốt tất cả các tầng (all layers) của mô hình GPT-2 XL. Bằng cách thiết lập vòng lặp phân tích qua từng `Transformer Block`, báo cáo này hướng dẫn cách trích xuất độ phân tán (Variance), giá trị trung bình (Means) và cấu hình lại ma trận Độ tương đồng Cosine (Cosine Similarity) cho các Token đích và Phi đích. Những kết quả thu được sẽ được trực quan hóa kết cấu theo độ sâu của kiến trúc mạng lưới.

---

## 1. Mở Đầu (Introduction)
Phân tích theo một tầng cố định (như layer-6 trước đó) cung cấp cái nhìn cục bộ, nhưng không diễn giải trọn vẹn "chu kỳ sống" của một mã token học thuật khi đi xuyên qua độ sâu của một LLM khổng lồ.
Thông qua thử thách lập trình này, ta sẽ:
- Thay vì nhỏ lẻ, phân rã mạng `GPT-2 XL` (có tới 48 transformer blocks và số chiều nhúng là 1600).
- Chạy hệ thống trên một trục tính toán hàng loạt (batch compute level), từ đó đánh giá sự thay đổi biểu diễn theo thời gian khi tiến dần về các tầng cận cuổi.
- Đối sánh mức độ đa dạng theo văn cảnh của nhóm tokens (Target vs. Non-target tokens) thông qua phương sai (variance) trên ma trận attention Q, K, V.

---

## 2. Phương Pháp Luận Và Giải Pháp Kỹ Thuật (Methodology)

### 2.1. Mã Hóa Hàm Khảo Sát Lớp Động (Dynamic Layer Scanning)
Sử dụng bộ công cụ PyTorch, ta xây dựng một hàm lặp để quét và trích xuất điểm kết nối:
1. Xác định vị trí Index của Token mục tiêu linh hoạt ứng với các câu có độ dài ngắn khác nhau.
2. Tại mỗi tầng `l` $(1 \le l \le 48)$, vector hàm kích hoạt tương ứng cho "Target" và một token ngẫu nhiên "Non-target" kế trước nó sẽ được tách bạch.
3. Kích thước mong đợi trong GPT-2 XL sau khi tách $Q, K, V$ sẽ là $\sim \text{Seq} \times 1600 \times 3$.

### 2.2. Đo Lường Phương Sai Nhóm (Variance Calculation)
Để đánh giá tác động của "context" lên cách thức mạng nơ-ron nhận thức chung một cụm từ ("her") trong 54 tình huống câu văn khác nhau:
- **Nguyên lý:** Nếu mô hình đối xử với từ "her" y hệt nhau dù nó đứng ở đâu, Phương sai sẽ $\approx 0$. Ngược lại, Phương sai mở rộng ám chỉ tầm ảnh hưởng rất lớn từ các chuỗi ngữ cảnh mồi.
- **Tính toán:** $V_{target} = \text{Var}(X_{layer=l, \space \text{token}="her"})\ \text{trên}\ 54 \text{ mẫu câu}$.

### 2.3. Tạo Ma Trận Khối Liên Hiệp (Cosine Matrix Block & Histogram Masking)
Tiếp tục ứng dụng Matrix Mask $(\text{size} = 4800 \times 4800)$ để bốc tách phần giao tuyến $Q-Q$, $K-K$ và rẽ nhánh của $Q-K, K-V$. Trích xuất biểu đồ phân phối Histogram từng tầng riêng rẽ rồi tổng hợp (Stack).

---

## 3. Khám Phá Các Tầng Mạng Ẩn (Analysis & Visualizations)

Việc ghim Plotting các phân phối Cosine xuyên không gian đa lớp mang về một góc nhìn thị giác giống quang phổ:

1. **Hiệu ứng thu hẹp phân cực (Convergence to Zero):**
   - Rất ấn tượng, ở các tầng nông (early layers), Cosine Similarity giữa các block có tính tụ tập rất mạnh bám sát miền hội tụ cao $(\approx 1.0/-1.0)$.
   - Càng trượt sâu xuống những block cuối (deeper into the model), phân bố bị là phẳng đi và thu trọng tâm dần về mức $0$.

2. **Lý giải về mặt Cơ Học (Mechanistic Reason):** 
   - Hiện tượng này phản chiếu bản chất của ngôn ngữ: Ở các tầng dưới, hệ thống mới chỉ "đọc và ghim" biểu diễn tĩnh ban đầu theo tự vựng của "her" (nên tương đồng cao). 
   - Đến các tầng trong cùng, mô hình dồn dập tích luỹ sự tập trung vào chức năng dự đoán từ ngữ đứng theo sau (subsequent prediction context). Vì các câu đa dạng đều có luồng văn cảnh cá biệt, các Vector mang "trách nhiệm tiếp theo" này sẽ phân huỷ dần sự giống nhau nguyên bản ban đầu. 

---

## 4. Kết Luận (Conclusion)
Thông qua thủ pháp quan sát toàn cục quy mô kiến trúc (across layers) trên siêu vi mô mô hình GPT-2 XL, chúng ta thấu thị được chặng hành trình sinh học của Attention. Tại đó, LLMs có vòng đời tự động chuyển hướng quy trình học: đi từ định hình đặc trưng ngữ nghĩa cơ sở (hiệu ứng liên cực lớn), dần hoà quyện theo phân hoá sự kiến giải ngữ cảnh để kết nối cấu trúc cho những token vô định ở tương lai (hiệu ứng suy tàn hội tụ). Khám phá này củng cố nền tảng diễn giải cơ học một cách sâu sắc và thực chứng.

---

## Tài Liệu Tham Khảo (Citations)
1. Dữ liệu trích xuất từ phần phụ đề và mã lệnh bài toán: `aero_LLM_03_CodeChallenge Token-related similarities across layers.md` (Giới thiệu các hàm tính Variance, Mean, Cosine Similarity và kỹ năng Stack Histogram cho GPT-2 XL).
