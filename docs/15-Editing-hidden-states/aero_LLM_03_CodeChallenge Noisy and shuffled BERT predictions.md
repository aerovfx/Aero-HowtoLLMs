# Thử thách Lập trình: Dự đoán BERT với Nhiễu và Hoán vị (Noisy and Shuffled BERT Predictions)

## Tóm tắt (Abstract)
Báo cáo này khảo sát tác động của hai loại can thiệp nhân quả lên mô hình BERT Large: Hoán vị ngẫu nhiên các chiều Embeddings (Shuffling) và Bơm nhiễu Gaussian (Noise Injection). Thông qua việc sử dụng Prompt nổi tiếng từ phim "Phù thủy xứ Oz", nghiên cứu định lượng mức độ sụp đổ của khả năng dự đoán từ bị che khuất (Masked token). Thực nghiệm cho thấy việc hoán vị chiều không gian dẫn đến sự sụp đổ hoàn toàn về ngữ nghĩa (hầu hết trả về dấu câu vô nghĩa), trong khi việc bơm nhiễu có kiểm soát (Scale $\sigma=2$) vẫn bảo toàn được khả năng sinh từ nhưng làm suy giảm độ chính xác và tăng dần sai số khi can thiệp ở các tầng sâu hơn.

---

## 1. Mở Đầu (Introduction)
Can thiệp nhân quả lên Hidden States không chỉ giới hạn ở việc thay đổi quy mô (Scaling) mà còn có thể tác động lên "Cấu trúc" (Structure) và "Độ trung thực" (Fidelity) của thông tin. Báo cáo này so sánh hai phương pháp phá hủy dữ liệu khác nhau để tìm hiểu xem mô hình nhạy cảm hơn với vị trí tọa độ của vector hay với độ lớn của tín hiệu nhiễu.

---

## 2. Phương Pháp Nghiên Cứu (Methodology)

### 2.1. Hoán vị Embeddings (Exercise 1 - 3)
- **Kỹ thuật:** Sử dụng `torch.randperm()` để tráo đổi thứ tự của các chiều thuộc embeddings dimension của một token duy nhất (Masked token). Đặc điểm của phương pháp này là bảo toàn giá trị cường độ nhưng phá hủy hoàn toàn ý nghĩa tọa độ ẩn.
- **Dữ liệu:** "Pay no attention to that man [MASK] the curtain". Target: "behind".

### 2.2. Bơm Nhiễu Gaussian Động (Exercise 4)
- **Kỹ thuật:** Thay vì hoán vị, ta cộng thêm nhiễu $\epsilon \sim \mathcal{N}(0, \sigma^2)$ vào vector nguyên bản.
- **Tinh chỉnh:** Giá trị $\sigma$ được tính toán động dựa trên độ lệch chuẩn thực tế của chính các vector tại tầng đó: `noise = torch.randn_like(hidden) * hidden.std() * scale_index`. Điều này đảm bảo nhiễu luôn có tỷ lệ tương xứng với cường độ tín hiệu nội tại của mô hình.

---

## 3. Kết Quả Và Thảo Luận (Results & Analysis)

### 3.1. Thảm họa Hoán vị (Catastrophic Shuffling)
Kết quả dự đoán của BERT sau khi bị hoán vị chiều tại bất kỳ tầng nào đều trở thành các chuỗi vô nghĩa như dấu phẩy, dấu chấm hoặc ký tự lạ. 
- **Nhận định:** Điều này chứng minh rằng "Ý nghĩa" (Meaning) trong LLM được mã hóa cực kỳ chặt chẽ vào các trục tọa độ cụ thể trong không gian embeddings. Việc xáo trộn chúng tương đương với việc phá hủy từ điển mã hóa của mô hình.

### 3.2. Sự Suy Giảm Có Quy Luật Của Nhiễu
- **Độ chính xác:** Khác với hoán vị, bơm nhiễu vẫn cho phép mô hình dự đoán ra "từ" (Words), dù đôi khi sai mục tiêu (vd: "in", "to" thay cho "behind").
- **Tác động theo tầng:** Sai số (Loss) và sự sụt giảm Log Softmax của từ đúng tăng dần theo độ sâu của tầng bị can thiệp. Việc bơm nhiễu ở lớp cuối cùng (Layer 23) gây ra tác động nặng nề nhất, do mô hình không còn các tầng hạ nguồn để "lọc" bớt nhiễu hoặc tái cấu trúc lại thông tin.

### 3.3. So sánh Phân phối: Nhiễu vs. Tín Hiệu (Exercise 5)
Phân tích Histogram cho thấy một thực tế quan trọng: Phân phối hoạt hóa thực của BERT tại các Transformer Blocks (Hidden States) cực kỳ hẹp và có đỉnh nhọn hơn nhiều so với phân phối Gaussian chuẩn.
- **Nghịch lý Gaussian:** Việc sử dụng nhiễu Gaussian để kiểm thử mô hình thực chất là đưa vào một loại nhiễu có đuôi (tails) rộng hơn nhiều so với phân phối tự nhiên của mô hình. Điều này đặt ra câu hỏi về tính hợp lệ của việc dùng phân phối chuẩn để mô phỏng "nhiễu sinh học/vật lý" trong các thí nghiệm Interpretability.

---

## 4. Kết Luận
Hoán vị chiều không gian là can thiệp mang tính hủy diệt, trong khi bơm nhiễu mang tính thống kê. Thí nghiệm khẳng định rằng LLM nhạy cảm nhất với các can thiệp ở những giai đoạn tính toán cuối cùng. Việc hiểu rõ sự khác biệt giữa hình dạng phân phối (Distribution shape) của nhiễu và tín hiệu thực là yếu tố then chốt để thiết kế các kịch bản can thiệp nhân quả chính xác hơn trong tương lai.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Noisy vs Shuffled BERT dựa trên tài liệu `aero_LLM_03_CodeChallenge Noisy and shuffled BERT predictions.md`. So sánh trực quan hóa giữa Histogram Gaussian và Hidden State thực tế.
