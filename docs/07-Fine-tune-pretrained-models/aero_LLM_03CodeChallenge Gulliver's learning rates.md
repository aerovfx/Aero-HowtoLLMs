
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
# Đánh giá Ảnh hưởng của Learning Rate trong Fine-tuning GPT-2 trên *Gulliver’s Travels*

## Tóm tắt (Abstract)

Bài viết này nghiên cứu tác động của các mức learning rate khác nhau trong quá trình fine-tuning mô hình GPT-2 trên văn bản *Gulliver’s Travels*. Thông qua thí nghiệm với ba learning rate (10⁻⁴, 10⁻⁵, 10⁻⁶), nghiên cứu đánh giá sự thay đổi của training loss và tỷ lệ token đặc trưng trong văn bản sinh ra. Kết quả cho thấy learning rate ảnh hưởng đáng kể đến mức độ thích nghi và nguy cơ overfitting của mô hình. Đồng thời, nghiên cứu nhấn mạnh rằng các chỉ số định lượng đơn thuần chưa đủ để đánh giá toàn diện chất lượng mô hình sinh ngôn ngữ.

---

## 1. Giới thiệu (Introduction)

Fine-tuning các mô hình ngôn ngữ lớn (LLMs) thường yêu cầu sử dụng learning rate nhỏ nhằm bảo toàn tri thức nền tảng đã được huấn luyện trước. Mục tiêu chính không phải là thay đổi hoàn toàn mô hình, mà chỉ “điều chỉnh nhẹ” để mô hình nhận diện tốt hơn các đặc trưng của dữ liệu mục tiêu .

Trong nghiên cứu này, một bài toán thực nghiệm được xây dựng nhằm đánh giá ảnh hưởng của learning rate đến hiệu quả fine-tuning GPT-2 trên *Gulliver’s Travels*. Thông qua việc lặp lại quy trình huấn luyện với nhiều learning rate, tác giả mong muốn làm rõ mối quan hệ giữa tốc độ học, mức độ thích nghi và khả năng tổng quát hóa.

---

## 2. Phương pháp (Methodology)

### 2.1. Thiết kế Thí nghiệm

Thí nghiệm được xây dựng dựa trên việc lặp lại quy trình fine-tuning với ba learning rate khác nhau:

* 10⁻⁴
* 10⁻⁵
* 10⁻⁶

Mỗi mô hình được huấn luyện từ cùng một bản GPT-2 pre-trained ban đầu nhằm đảm bảo tính công bằng trong so sánh .

Số mẫu huấn luyện tiêu chuẩn được đặt là 800, tuy nhiên có thể giảm xuống 20–50 mẫu trong giai đoạn thử nghiệm nhằm kiểm tra lỗi chương trình.

---

### 2.2. Tiền xử lý và Phân tích Token

Văn bản *Gulliver’s Travels* được token hóa và thống kê 100 token xuất hiện nhiều nhất. Danh sách này đóng vai trò làm tập đặc trưng để đánh giá mức độ “học phong cách” của mô hình.

Một hàm đánh giá được xây dựng để tính tỷ lệ token sinh ra thuộc nhóm 100 token phổ biến này .

---

### 2.3. Xây dựng Hàm Đánh giá

Hai hàm chính được thiết kế:

#### (1) Hàm đếm token phổ biến

Hàm này:

* Nhận mô hình làm đầu vào
* Sinh văn bản từ token ngẫu nhiên
* Tính tỷ lệ token trùng với danh sách 100 token phổ biến

Hàm cho phép đánh giá linh hoạt nhiều mô hình khác nhau .

#### (2) Hàm huấn luyện và đánh giá

Hàm này nhận hai tham số:

* Learning rate
* Số mẫu huấn luyện

Bên trong hàm:

1. Tải mô hình GPT-2 mới
2. Đánh giá trước huấn luyện
3. Tiến hành fine-tuning
4. Đánh giá sau huấn luyện
5. Xuất training loss và kết quả đánh giá

Cách tiếp cận này cho phép so sánh trực tiếp tác động của learning rate .

---

### 2.4. Trực quan hóa Dữ liệu

Trong bài tập thứ ba, kết quả được biểu diễn bằng:

* Biểu đồ đường: training loss
* Biểu đồ cột: tỷ lệ token đặc trưng

Do số lượng điểm dữ liệu lớn, chỉ một phần các marker được hiển thị để đảm bảo tính trực quan .

---

## 3. Kết quả Thực nghiệm (Experimental Results)

### 3.1. Ảnh hưởng của Learning Rate đến Training Loss

Kết quả cho thấy:

* Learning rate cao hơn → loss giảm nhanh hơn
* Learning rate thấp → loss giảm chậm, ít biến động

Mô hình với learning rate 10⁻⁴ đạt loss thấp nhất, cho thấy khả năng thích nghi mạnh mẽ nhất .

Tuy nhiên, loss thấp không đồng nghĩa với chất lượng mô hình tốt hơn trong mọi trường hợp.

---

### 3.2. Tỷ lệ Token Đặc trưng

Tất cả các mô hình sau fine-tuning đều cho thấy sự gia tăng tỷ lệ token đặc trưng:

* Trước huấn luyện: tương đương nhau
* Sau huấn luyện: đều tăng lên

Mô hình có learning rate lớn đạt khoảng 60–61%, trong khi learning rate nhỏ chỉ đạt khoảng 52% .

Điều này phản ánh mức độ thích nghi với văn bản mục tiêu.

---

### 3.3. So sánh Ba Mô hình

| Learning Rate | Training Loss | Token Tỷ lệ | Mức độ Thích nghi |
| ------------- | ------------- | ----------- | ----------------- |
| 10⁻⁴          | Thấp nhất     | Cao nhất    | Rất mạnh          |
| 10⁻⁵          | Trung bình    | Cao         | Cân bằng          |
| 10⁻⁶          | Cao nhất      | Thấp hơn    | Nhẹ nhàng         |

Mỗi learning rate thể hiện một chiến lược fine-tuning khác nhau.

---

## 4. Thảo luận (Discussion)

### 4.1. Mối quan hệ giữa Loss và Overfitting

Loss tiến gần về 0 phản ánh mức độ ghi nhớ cao đối với dữ liệu huấn luyện. Tuy nhiên, điều này có thể dẫn đến:

* Xóa bỏ tri thức nền tảng
* Giảm khả năng tổng quát hóa
* Tăng nguy cơ sao chép nội dung gốc

Việc fine-tuning quá mạnh có thể làm mất đi tính “đa năng” của mô hình .

---

### 4.2. Giới hạn của Chỉ số Định lượng

Hai chỉ số chính được sử dụng là:

* Training loss
* Tỷ lệ token phổ biến

Mặc dù hữu ích, chúng không phản ánh đầy đủ:

* Độ mạch lạc
* Tính sáng tạo
* Độ phù hợp ngữ cảnh
* Giá trị ứng dụng thực tế

Do đó, chỉ số định lượng cần được kết hợp với đánh giá định tính từ con người .

---

### 4.3. Lựa chọn Learning Rate Tối ưu

Không tồn tại learning rate “tốt nhất” trong mọi trường hợp.

* Ứng dụng chuyên biệt → learning rate cao
* Ứng dụng tổng quát → learning rate thấp hoặc trung bình

Việc lựa chọn phụ thuộc vào mục tiêu sử dụng mô hình.

---

## 5. Kết luận (Conclusion)

Nghiên cứu cho thấy learning rate có vai trò quyết định trong quá trình fine-tuning GPT-2:

1. Learning rate cao giúp mô hình học nhanh nhưng dễ overfitting.
2. Learning rate thấp bảo toàn tri thức nền nhưng thích nghi chậm.
3. Chỉ số định lượng chưa đủ để đánh giá chất lượng mô hình sinh văn bản.

Do đó, fine-tuning hiệu quả đòi hỏi sự cân bằng giữa:

* Tốc độ học
* Mức độ thích nghi
* Khả năng tổng quát hóa
* Đánh giá định tính

Fine-tuning nên được xem là quá trình “tinh chỉnh nhẹ” thay vì tái huấn luyện toàn diện .

---

## Tài liệu Tham khảo (References)

* *3 - CodeChallenge Gulliver's learning rates.txt*
  Nguồn tài liệu nội bộ do người dùng cung cấp.

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
| [Tinh Chỉnh Mô Hình BERT Cho Bài Toán Phân Loại Cảm Xúc Văn Bản IMDb](aero_LLM_015_Fine-tuning BERT for classification.md) | [Xem bài viết →](aero_LLM_015_Fine-tuning BERT for classification.md) |
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
| 📌 **[Đánh giá Ảnh hưởng của Learning Rate trong Fine-tuning GPT-2 trên *Gulliver’s Travels*](aero_LLM_03CodeChallenge Gulliver's learning rates.md)** | [Xem bài viết →](aero_LLM_03CodeChallenge Gulliver's learning rates.md) |
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
