
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [07 Fine tune pretrained models](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01-LLM_Course/index.md)
- [🔢 Module 02: Tokenization](../../02-Words-to-tokens-to-numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04-buildGPT/index.md)
- [🎯 Module 07: Fine-tuning](../../07-Fine-tune-pretrained-models/index.md)
- [🔍 Module 19: AI Safety](../../19-AI-safety/index.md)
- [🐍 Module 20: Python for AI](../../20-Python-Colab-notebooks/index.md)
---
<!-- Aero-Navigation-End -->
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
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [📂 Module: 07-Fine-tune-pretrained-models](README.md) | [Xem bài viết →](README.md) |
| [Fine-tuning Có Mục Tiêu và Đóng Băng Chính Xác Trọng Số Trong Mô Hình Ngôn Ngữ Lớn](aero_LLM_010_CodeChallenge Fine-tuning and targeted freezing (part 1).md) | [Xem bài viết →](aero_LLM_010_CodeChallenge Fine-tuning and targeted freezing (part 1).md) |
| [Phân Tích Hiệu Quả Fine-tuning và Targeted Freezing (Phần 2): Đánh Giá Bằng Trực Quan Hóa và Chuẩn Ma Trận](aero_LLM_011_CodeChallenge Fine-tuning and targeted freezing (part 2).md) | [Xem bài viết →](aero_LLM_011_CodeChallenge Fine-tuning and targeted freezing (part 2).md) |
| [Fine-tuning Hiệu Quả Tham Số (Parameter-Efficient Fine-Tuning – PEFT) Trong Mô Hình Ngôn Ngữ Lớn](aero_LLM_012_Parameter-efficient fine-tuning (PEFT).md) | [Xem bài viết →](aero_LLM_012_Parameter-efficient fine-tuning (PEFT).md) |
| [Mô Hình CodeGen Cho Bài Toán Hoàn Thành Mã Nguồn: Kiến Trúc, Huấn Luyện và Ứng Dụng](aero_LLM_013_CodeGen for code completion.md) | [Xem bài viết →](aero_LLM_013_CodeGen for code completion.md) |
| [Fine-tuning Mô Hình CodeGen Cho Bài Toán Giải Tích: Phương Pháp, Đánh Giá và Ứng Dụng](aero_LLM_014_CodeChallenge Fine-tune codeGen for calculus.md) | [Xem bài viết →](aero_LLM_014_CodeChallenge Fine-tune codeGen for calculus.md) |
| 📌 **[Tinh Chỉnh Mô Hình BERT Cho Bài Toán Phân Loại Cảm Xúc Văn Bản IMDb](aero_LLM_015_Fine-tuning BERT for classification.md)** | [Xem bài viết →](aero_LLM_015_Fine-tuning BERT for classification.md) |
| [📘 Ứng Dụng Mô Hình BERT Trong Phân Tích Cảm Xúc Đánh Giá Phim IMDB](aero_LLM_016_CodeChallenge IMDB sentiment analysis using BERT.en_US.md) | [Xem bài viết →](aero_LLM_016_CodeChallenge IMDB sentiment analysis using BERT.en_US.md) |
| [📘 Ứng Dụng Gradient Clipping và Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu](aero_LLM_017_Gradient clipping and learning rate scheduler (part 1).en_US.md) | [Xem bài viết →](aero_LLM_017_Gradient clipping and learning rate scheduler (part 1).en_US.md) |
| [📘 Phân Tích Learning Rate Scheduler Trong Huấn Luyện Mô Hình Học Sâu Quy Mô Lớn](aero_LLM_018_Gradient clipping and learning rate scheduler (part 2).md) | [Xem bài viết →](aero_LLM_018_Gradient clipping and learning rate scheduler (part 2).md) |
| [📘 Kết Hợp Gradient Clipping, Freezing và Learning Rate Scheduler Trong Fine-Tuning Mô Hình BERT](aero_LLM_019_CodeChallenge Clip, freeze, and schedule BERT.md) | [Xem bài viết →](aero_LLM_019_CodeChallenge Clip, freeze, and schedule BERT.md) |
| [Tối Ưu Hóa Quá Trình Tiền Huấn Luyện Mô Hình Ngôn Ngữ Lớn: Phân Tích Các Chiến Lược Tính Toán và Học Tập](aero_LLM_01_What does fine-tuning mean.md) | [Xem bài viết →](aero_LLM_01_What does fine-tuning mean.md) |
| [Lưu Trữ và Tải Lại Mô Hình Học Sâu Trong PyTorch và Hugging Face: Phương Pháp, Cấu Trúc và Đánh Giá](aero_LLM_020_Saving and loading trained models.md) | [Xem bài viết →](aero_LLM_020_Saving and loading trained models.md) |
| [Ứng Dụng Mô Hình BERT Trong Phân Loại Văn Bản Văn Học: Trường Hợp Alice và Edgar](aero_LLM_021_BERT decides Alice or Edgar.md) | [Xem bài viết →](aero_LLM_021_BERT decides Alice or Edgar.md) |
| [Đồng Tiến Hóa Mô Hình Sinh Văn Bản và Mô Hình Phân Loại: Trường Hợp Alice và Edgar](aero_LLM_022_CodeChallenge Evolution of Alice and Edgar (part 1).md) | [Xem bài viết →](aero_LLM_022_CodeChallenge Evolution of Alice and Edgar (part 1).md) |
| [📘 Đánh Giá Mô Hình Sinh Văn Bản Thông Qua Phân Loại BERT: Nghiên Cứu Trường Hợp Alice và Edgar](aero_LLM_023_CodeChallenge Evolution of Alice and Edgar (part 2).md) | [Xem bài viết →](aero_LLM_023_CodeChallenge Evolution of Alice and Edgar (part 2).md) |
| [Fine-tuning Mô hình GPT-2 trên Tác phẩm *Gulliver’s Travels*: Phân tích Thực nghiệm và Đánh giá Hiệu quả](aero_LLM_02_Fine-tune a pretrained GPT2.md) | [Xem bài viết →](aero_LLM_02_Fine-tune a pretrained GPT2.md) |
| [Đánh giá Ảnh hưởng của Learning Rate trong Fine-tuning GPT-2 trên *Gulliver’s Travels*](aero_LLM_03CodeChallenge Gulliver's learning rates.md) | [Xem bài viết →](aero_LLM_03CodeChallenge Gulliver's learning rates.md) |
| [Nghiên cứu Quy trình Sinh Văn bản từ Mô hình Ngôn ngữ Tiền Huấn luyện GPT-2](aero_LLM_04_On generating text from pretrained models.md) | [Xem bài viết →](aero_LLM_04_On generating text from pretrained models.md) |
| [Tinh Chỉnh Mô Hình GPT-2 Bằng Hàm Mất Mát KL Divergence Để Tối Ưu Hóa Việc Sinh Token Chứa Ký Tự “X”](aero_LLM_05_CodeChallenge Maximize the X factor..md) | [Xem bài viết →](aero_LLM_05_CodeChallenge Maximize the X factor..md) |
| [Tinh Chỉnh Mô Hình GPT-Neo Để Mô Phỏng Phong Cách Văn Học Alice in Wonderland và Edgar Allan Poe](aero_LLM_06_Alice in Wonderland and Edgar Allen Poe (with GPT-neo).md) | [Xem bài viết →](aero_LLM_06_Alice in Wonderland and Edgar Allen Poe (with GPT-neo).md) |
| [Đánh Giá Định Lượng và Định Tính Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp Văn Phong *Alice* và *Edgar Allan Poe*](aero_LLM_07_CodeChallenge Quantify the AliceEdgar fine-tunin.md) | [Xem bài viết →](aero_LLM_07_CodeChallenge Quantify the AliceEdgar fine-tunin.md) |
| [Định Lượng Hiệu Quả Tinh Chỉnh Phong Cách Văn Học: Thử Thách Alice và Edgar](aero_LLM_07_CodeChallenge Quantify the AliceEdgar fine-tuning.md) | [Xem bài viết →](aero_LLM_07_CodeChallenge Quantify the AliceEdgar fine-tuning.md) |
| [Mô Phỏng Hội Thoại Giữa Hai Mô Hình Ngôn Ngữ Sau Fine-tuning: Trường Hợp *Alice* và *Edgar*](aero_LLM_08_CodeChallenge A chat between Alice and Edgar.md) | [Xem bài viết →](aero_LLM_08_CodeChallenge A chat between Alice and Edgar.md) |
| [Tinh Chỉnh Từng Phần Bằng Cách Đóng Băng Trọng Số Attention: Chiến Lược Tối Ưu Hóa Tham Số Cho LLM](aero_LLM_09_Partial fine-tuning by freezing attention weights.md) | [Xem bài viết →](aero_LLM_09_Partial fine-tuning by freezing attention weights.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
