
# 📘 Phương Pháp Lấy Mẫu Ngẫu Nhiên và Huấn Luyện Mô Hình GPT-2 Thu Gọn: Phân Tích Thực Nghiệm Với Dữ Liệu Văn Bản Cổ Điển

---

## Tóm tắt (Abstract)

Huấn luyện các mô hình ngôn ngữ dựa trên Transformer đòi hỏi sự kết hợp giữa kiến trúc phù hợp, phương pháp lấy mẫu dữ liệu hiệu quả và hạ tầng tính toán mạnh mẽ. Bài viết này phân tích phương pháp huấn luyện một biến thể của mô hình GPT-2 (Model 5) với kỹ thuật lấy mẫu ngẫu nhiên trực tiếp từ văn bản, thay vì sử dụng DataLoader truyền thống. Dựa trên tài liệu *CodeChallenge: Train Model 5 with Modifications*, nghiên cứu đánh giá hiệu quả của phương pháp sampling, vai trò của GPU, và khả năng học cấu trúc ngôn ngữ từ dữ liệu hạn chế. Kết quả cho thấy mô hình có thể học được các đặc trưng hình thức của văn bản chỉ với số lượng mẫu huấn luyện rất nhỏ. 

---

## 1. Giới thiệu (Introduction)

Trong huấn luyện mô hình học sâu, quy trình cơ bản gồm thu thập dữ liệu, tiền xử lý, huấn luyện và đánh giá thường được lặp lại với cấu trúc tương tự. Theo tài liệu, phần lớn các pipeline huấn luyện đều có kiến trúc gần giống nhau, chỉ khác ở dữ liệu, kiến trúc mô hình và hàm mất mát. 

Trong bài thực hành này, tác giả giới thiệu:

* Một phương pháp lấy mẫu dữ liệu thay thế,
* Cách áp dụng cho mô hình dựa trên GPT-2,
* Thực nghiệm trên văn bản *Gulliver’s Travels*,
* So sánh hiệu suất CPU và GPU.

Mục tiêu của bài viết là phân tích tác động của các yếu tố trên đến quá trình huấn luyện.

---

## 2. Cơ Sở Lý Thuyết (Theoretical Background)

### 2.1. Huấn Luyện Mô Hình Ngôn Ngữ Dựa Trên Transformer

GPT-2 là mô hình ngôn ngữ tự hồi quy sử dụng kiến trúc Transformer Decoder. Nó học phân phối xác suất:

[
P(x_t | x_1, x_2, \dots, x_{t-1})
]

Trong đó, mỗi token được dự đoán dựa trên ngữ cảnh trước đó.

Model 5 trong nghiên cứu này là phiên bản rút gọn của GPT-2 với các tham số tương đương bản 124M. 

---

### 2.2. Sampling Trong Huấn Luyện Ngôn Ngữ

Thông thường, dữ liệu được nạp thông qua `Dataset` và `DataLoader`. Tuy nhiên, tài liệu đề xuất phương pháp lấy mẫu trực tiếp từ vector token, không cần xây dựng lớp dataset riêng. 

Cách tiếp cận này giúp:

* Giảm độ phức tạp code,
* Tăng tính linh hoạt,
* Phù hợp cho thử nghiệm nhanh.

---

## 3. Phương Pháp Thực Nghiệm (Methodology)

### 3.1. Dữ Liệu và Tokenization

Nguồn dữ liệu là tác phẩm *Gulliver’s Travels* từ Project Gutenberg. Văn bản được:

1. Tải về tự động,
2. Làm sạch,
3. Token hóa bằng tokenizer GPT-2.

Sau xử lý, tập dữ liệu gồm khoảng 158.000 token. 

---

### 3.2. Phân Chia Dữ Liệu

Dữ liệu được chia theo tỷ lệ:

* 90% cho huấn luyện,
* 10% cho kiểm thử.

Phần test luôn nằm ở cuối văn bản, điều này có thể gây sai lệch do đặc trưng nội dung cuối sách khác đầu sách. 

---

### 3.3. Hàm Lấy Mẫu Ngẫu Nhiên

Thay vì DataLoader, một hàm sampling được xây dựng như sau:

1. Chọn ngẫu nhiên các vị trí bắt đầu,
2. Tạo tensor chỉ số,
3. Truy xuất token,
4. Sinh input–target với độ trễ một token.

Kết quả có dạng:

[
X, Y \in \mathbb{R}^{B \times T}
]

Trong đó:

* (B): batch size,
* (T = 256): sequence length.



---

### 3.4. Cấu Trúc Input–Target

Target được dịch sang phải một token so với input:

[
Y_i = X_{i+1}
]

Điều này phù hợp với bài toán language modeling tự hồi quy. 

---

## 4. Thiết Lập Huấn Luyện (Training Setup)

### 4.1. Hạ Tầng Phần Cứng

Do quy mô mô hình lớn, GPU được sử dụng bắt buộc. Theo tài liệu, chạy trên CPU mất khoảng 45 giây cho một forward pass, trong khi GPU chỉ mất khoảng 1 giây. 

---

### 4.2. Kiến Trúc Mô Hình

Model 5 dựa trên GPT-2 small với:

* 12 Transformer blocks,
* 768 hidden units,
* 12 attention heads.

Các tham số được sao chép từ cấu hình GPT-2 gốc. 

---

### 4.3. Vòng Lặp Huấn Luyện

Mô hình được huấn luyện bằng cách lấy 500 batch ngẫu nhiên:

* Mỗi batch độc lập,
* Không duyệt toàn bộ dữ liệu,
* Không khái niệm epoch truyền thống.

Điều này khác với huấn luyện qua DataLoader tuần tự. 

---

## 5. Đánh Giá Mô Hình (Evaluation)

### 5.1. Đánh Giá Định Lượng

Loss được tính bằng cross-entropy và được ghi lại sau mỗi 100 batch. Biểu đồ loss cho thấy:

* Giá trị ban đầu rất cao,
* Giảm nhanh trong vài trăm bước,
* Hội tụ sớm.



---

### 5.2. Đánh Giá Định Tính

Trước huấn luyện, mô hình sinh chuỗi lặp vô nghĩa:

```
ions ions ions ions...
```

Sau huấn luyện, văn bản sinh ra có:

* Dòng mới,
* Dấu câu,
* Cấu trúc câu đơn giản.

Mặc dù chưa có ngữ nghĩa rõ ràng, hình thức văn bản đã được học. 

---

### 5.3. Học Cấu Trúc Văn Bản

Mô hình nhanh chóng học được:

* Khoảng cách dòng,
* Vị trí xuống dòng,
* Mẫu định dạng văn bản.

Điều này cho thấy Transformer có khả năng trích xuất cấu trúc hình thức rất nhanh. 

---

## 6. Thảo Luận (Discussion)

### 6.1. Ưu Điểm Của Random Sampling

Phương pháp lấy mẫu ngẫu nhiên mang lại:

* Code gọn nhẹ,
* Tốc độ phát triển nhanh,
* Phù hợp cho thử nghiệm.

Nó đặc biệt hữu ích trong giai đoạn prototyping. 

---

### 6.2. Rủi Ro Của Sampling

Tuy nhiên, phương pháp này có nguy cơ:

* Lặp lại dữ liệu,
* Bỏ sót một số vùng văn bản,
* Phân phối không đồng đều.

Điều này có thể ảnh hưởng đến khả năng tổng quát hóa. 

---

### 6.3. So Sánh Với Dữ Liệu Y Sinh

Tài liệu chỉ ra rằng rủi ro sampling ít nghiêm trọng với dữ liệu văn bản, do nguồn dữ liệu phong phú, nhưng rất nguy hiểm trong lĩnh vực y sinh, nơi dữ liệu hạn chế. 

---

### 6.4. Khả Năng Học Nhanh Của LLM

Một phát hiện quan trọng là:

> Chỉ với vài trăm batch, mô hình đã học được “hình dáng” của ngôn ngữ.

Điều này phản ánh sức mạnh biểu diễn của Transformer. 

---

## 7. Hạn Chế (Limitations)

Một số hạn chế của nghiên cứu:

* Không dùng full epoch,
* Test set có bias vị trí,
* Không so sánh với DataLoader chuẩn,
* Chỉ thử trên một corpus.

Do đó, kết quả mang tính minh họa hơn là tổng quát.

---

## 8. Kết Luận (Conclusion)

Bài viết đã phân tích phương pháp huấn luyện Model 5 với sampling ngẫu nhiên và các chỉnh sửa kỹ thuật. Các kết luận chính gồm:

1. Random sampling giúp đơn giản hóa pipeline huấn luyện.
2. GPU là yếu tố then chốt cho mô hình lớn.
3. Mô hình học rất nhanh cấu trúc văn bản.
4. Số lượng dữ liệu nhỏ vẫn tạo ra hiệu ứng học rõ rệt.
5. Phương pháp phù hợp cho nghiên cứu thử nghiệm.

Nghiên cứu khẳng định rằng ngay cả với dữ liệu hạn chế và pipeline đơn giản, Transformer vẫn thể hiện khả năng học mạnh mẽ.

---

## Tài Liệu Tham Khảo (References)

[1] CodeChallenge: Train Model 5 with Modifications, Lecture Transcript.


---
