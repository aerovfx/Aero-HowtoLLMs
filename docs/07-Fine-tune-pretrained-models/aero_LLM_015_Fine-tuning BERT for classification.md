
# Tinh Chỉnh Mô Hình BERT Cho Bài Toán Phân Loại Cảm Xúc Văn Bản IMDb

## Tóm tắt

Tinh chỉnh các mô hình ngôn ngữ đã được huấn luyện trước đang trở thành phương pháp chủ đạo trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP). Bài báo này trình bày phương pháp áp dụng mô hình BERT cho bài toán phân loại cảm xúc nhị phân dựa trên dữ liệu đánh giá phim từ IMDb. Nghiên cứu mô tả kiến trúc mô hình, quy trình tiền xử lý dữ liệu, chiến lược token hóa, phương pháp huấn luyện và đánh giá hiệu quả mô hình. Kết quả cho thấy phương pháp học chuyển giao giúp nâng cao độ chính xác ngay cả khi tài nguyên tính toán hạn chế. 

---

## Từ khóa

BERT, Fine-tuning, Phân tích cảm xúc, Học chuyển giao, Xử lý ngôn ngữ tự nhiên, IMDb

---

## 1. Giới thiệu

Trong những năm gần đây, các mô hình học sâu đã mang lại bước tiến lớn trong việc hiểu ngôn ngữ tự nhiên. Các mô hình được huấn luyện trước như BERT cho phép thích nghi nhanh chóng với các bài toán cụ thể thông qua kỹ thuật tinh chỉnh.

Phân tích cảm xúc là một trong những bài toán cơ bản của NLP, nhằm xác định thái độ tích cực hay tiêu cực trong văn bản. Trong nghiên cứu này, chúng tôi áp dụng BERT để phân loại các bài đánh giá phim thành hai nhóm: tích cực và tiêu cực. 

---

## 2. Các nghiên cứu liên quan

BERT sử dụng cơ chế tự chú ý hai chiều để học biểu diễn ngữ cảnh của văn bản. Nhiều nghiên cứu đã chứng minh rằng việc tinh chỉnh BERT mang lại hiệu quả cao trong các bài toán phân loại, hỏi đáp và truy xuất thông tin.

Học chuyển giao trong NLP giúp giảm đáng kể chi phí huấn luyện bằng cách tận dụng tri thức đã học từ các tập dữ liệu lớn. Nghiên cứu này kế thừa hướng tiếp cận đó. 

---

## 3. Phương pháp nghiên cứu

### 3.1 Kiến trúc mô hình

Mô hình được đề xuất gồm hai thành phần chính:

* Bộ mã hóa BERT đã huấn luyện sẵn
* Lớp phân loại tuyến tính

Đầu ra của BERT có kích thước 768 chiều, sau đó được đưa qua lớp dropout và lớp fully-connected để ánh xạ về 2 nhãn phân loại. 

Công thức phân loại:

[
y = \text{Softmax}(W h + b)
]

Trong đó (h) là vector đặc trưng từ BERT.

---

### 3.2 Tập dữ liệu

Tập dữ liệu IMDb gồm 50.000 bài đánh giá phim, được gán nhãn:

* 0: Tiêu cực
* 1: Tích cực

Chia thành:

* Tập huấn luyện: 25.000 mẫu
* Tập kiểm tra: 25.000 mẫu

Phần dữ liệu không giám sát không được sử dụng. Một tập con cân bằng được trích xuất để giảm thời gian huấn luyện. 

---

### 3.3 Tiền xử lý dữ liệu

#### 3.3.1 Token hóa

Văn bản được token hóa bằng tokenizer của BERT, tạo ra:

* Input IDs
* Attention Mask
* Token Type IDs

Chuỗi được:

* Cắt ngắn tối đa 512 token
* Đệm bằng số 0 nếu thiếu

Nhằm đảm bảo kích thước thống nhất trong mỗi batch. 

---

#### 3.3.2 Ánh xạ dữ liệu

Hàm tiền xử lý được áp dụng lên toàn bộ tập dữ liệu thông qua hàm `map`. Kết quả gồm:

* input_ids
* attention_mask
* labels

Cột văn bản gốc được loại bỏ để tiết kiệm bộ nhớ. 

---

### 3.4 Quy trình huấn luyện

#### 3.4.1 Bộ nạp dữ liệu

Sử dụng DataLoader của PyTorch với:

* Batch size: 32
* Xáo trộn ngẫu nhiên
* Chuyển sang tensor

Giúp huấn luyện hiệu quả trên GPU. 

---

#### 3.4.2 Tối ưu hóa

Quá trình huấn luyện sử dụng:

* AdamW Optimizer
* Cross-Entropy Loss
* Dropout = 0.1

Hàm mất mát phù hợp cho bài toán phân loại nhị phân. 

---

#### 3.4.3 Bước huấn luyện

Mỗi vòng lặp gồm:

1. Đưa dữ liệu lên GPU
2. Lan truyền xuôi
3. Tính loss
4. Lan truyền ngược
5. Cập nhật tham số

Nhãn dự đoán được xác định bằng giá trị logit lớn nhất. 

---

### 3.5 Đánh giá mô hình

Độ chính xác được tính theo công thức:

[
Accuracy = \frac{Số\ mẫu\ dự\ đoán\ đúng}{Tổng\ số\ mẫu}
]

Mô hình chưa huấn luyện cho độ chính xác xấp xỉ 50%, tương đương đoán ngẫu nhiên. Điều này cho thấy pipeline được xây dựng đúng. 

---

## 4. Kết quả thực nghiệm

Kết quả ban đầu cho thấy:

* Mô hình chưa huấn luyện: ~50% accuracy
* Sau tinh chỉnh: độ chính xác tăng rõ rệt
* Dữ liệu cân bằng giúp giảm sai lệch
* Padding và truncation ổn định quá trình học

Quy trình tiền xử lý đóng vai trò then chốt trong hiệu năng mô hình. 

---

## 5. Thảo luận

### 5.1 Ưu điểm

* Hiệu quả cao với dữ liệu nhỏ
* Thời gian huấn luyện ngắn
* Khả năng tổng quát tốt
* Dễ mở rộng cho nhiều bài toán

---

### 5.2 Hạn chế

* Phụ thuộc thư viện bên thứ ba
* Dễ xảy ra xung đột phiên bản
* Tốn bộ nhớ
* Khó giải thích kết quả

Các vấn đề về môi trường Python vẫn là thách thức phổ biến. 

---

## 6. Kết luận

Nghiên cứu đã xây dựng thành công mô hình BERT tinh chỉnh cho bài toán phân loại cảm xúc phim. Việc kết hợp mô hình nền tảng với lớp phân loại đơn giản giúp đạt hiệu quả cao và tiết kiệm tài nguyên.

Trong tương lai, có thể mở rộng sang:

* Phân loại đa lớp
* Thích nghi miền dữ liệu
* Nén mô hình
* Phân tích khả năng giải thích

---

## Tài liệu tham khảo

1. Tài liệu học tập: *Fine-tuning BERT for Classification*, “15 - Fine-tuning BERT for classification.en_US.txt”. 

---

