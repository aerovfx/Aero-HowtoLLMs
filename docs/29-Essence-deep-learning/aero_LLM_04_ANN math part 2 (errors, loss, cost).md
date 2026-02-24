# Học sâu: Giải tích ANN Phần 2 – Sai số, Mất mát và Chi phí (Errors, Loss, Cost)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về các cơ chế định lượng sai số trong mạng nơ-ron nhân tạo, đóng vai trò là "la bàn" để điều hướng quá trình học tập. chúng ta phân tích sự khác biệt giữa dự đoán của mô hình ($\hat{y}$) và giá trị thực tế ($y$), từ đó định nghĩa các hàm mất mát (loss functions) cho dữ liệu liên tục và rời rạc. Nghiên cứu thực hiện phân biệt giữa khái niệm "mất mát" (loss) trên từng mẫu đơn lẻ và "chi phí" (cost) trên toàn bộ tập dữ liệu, đồng thời thuyết minh về lý do tại sao việc tối ưu hóa hàm chi phí là mục tiêu tối thượng của mọi quy trình huấn luyện học sâu.

---

## 1. Định lượng Sai số (Quantifying Error)

Trong học sâu, sai số là khoảng cách giữa kỳ vọng và thực tế:
- **Dự đoán ($\hat{y}$):** Kết quả mà mô hình đưa ra (ví dụ: xác suất 98% là ảnh con mèo).
- **Thực tế ($y$):** Giá trị mục tiêu (target) đo lường được từ thế giới thực (ví dụ: thực tế là ảnh con chó, giá trị 0).
- **Phân loại sai số:**
    - **Sai số liên tục:** Dùng để dạy mô hình, có độ nhạy cao với các thay đổi nhỏ.
    - **Sai số nhị phân (Binarized):** Dùng để đánh giá hiệu năng (Accuracy), dễ hiểu nhưng kém nhạy bén trong quá trình tối ưu hóa.

---

## 2. Các Hàm Mất mát Chủ chốt (Loss Functions)

Mỗi loại bài toán đòi hỏi một thước đo sai số khác nhau:

### 2.1. Sai số Bình phương Trung bình (Mean Squared Error - MSE)
- **Ứng dụng:** Dùng cho dự đoán giá trị số liên tục (ví dụ: giá nhà, nhiệt độ).
- **Công thức:** $L = \frac{1}{2}(\hat{y} - y)^2$
- **Đặc điểm:** Việc bình phương giúp loại bỏ dấu âm và tạo ra một hàm lồi (convex) thuận lợi cho việc tính đạo hàm. Hệ số $1/2$ giúp triệt tiêu số dư khi tính đạo hàm đa thức.

### 2.2. Entropy chéo (Cross-Entropy)
- **Ứng dụng:** Dùng cho dự đoán phân loại nhị phân hoặc đa lớp (ví dụ: xác suất mắc bệnh).
- **Công thức:** $L = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$
- **Đặc điểm:** Phạt nặng những dự đoán sai với độ tự tin cao. Dấu âm giúp chuyển đổi các giá trị logarit âm thành một giá trị mất mát dương dễ diễn giải.

---

## 3. Từ Mất mát đến Hàm Chi phí (Cost Function)

Một sự nhầm lẫn phổ biến là coi Loss và Cost là một, nhưng chúng có sự khác biệt về quy mô:
- **Loss (Mất mát):** Tính trên **một mẫu** dữ liệu duy nhất.
- **Cost (Chi phí - $J$):** Là **trung bình cộng** của tất cả các giá trị Loss trên toàn bộ tập dữ liệu (hoặc một lô dữ liệu - batch).
  $$J(w) = \frac{1}{N} \sum_{i=1}^{N} L_i$$
Việc tối ưu hóa dựa trên Cost giúp mô hình có cái nhìn tổng quát về toàn bộ dữ liệu, tránh hiện tượng quá khớp (overfitting) nếu chỉ nhìn vào từng mẫu riêng lẻ.

---

## 4. Mục tiêu của Huấn luyện (Optimization Goal)

Toàn bộ quá trình huấn luyện có thể tóm gọn trong một biểu thức toán học duy nhất:
$$\min_{W} J(W)$$
Tìm tập hợp các trọng số $W$ sao cho hàm chi phí $J$ đạt giá trị nhỏ nhất. Lúc này, dự đoán của mô hình sẽ khớp nhất với thực tế. Trong thực tế, chúng ta thường sử dụng các "lô" (batches) nhỏ dữ liệu để tính toán trung bình chi phí, giúp cân bằng giữa tốc độ tính toán và độ chính xác của gradient.

---

## 5. Kết luận
Hiểu về sai số không chỉ là biết mô hình sai bao nhiêu, mà là biết cách chuyển hóa cái sai đó thành một hàm số có thể tối ưu hóa được. Hàm MSE và Cross-Entropy là nền tảng của hầu hết các kiến trúc AI hiện đại, từ các bộ phân loại đơn giản đến những hệ thống LLM phức tạp. Thấu hiểu mối quan hệ giữa dự đoán ($\hat{y}$) và mục tiêu ($y$) thông qua lăng kính của hàm chi phí chính là bước đệm then chốt để bước vào thế giới của lan truyền ngược (backpropagation) – "động cơ" thực sự giúp máy tính học tập.

---

## Tài liệu tham khảo (Citations)
1. Cơ chế định lượng sai số và các loại hàm mất mát dựa trên `aero_LL_04_ANN math part 2 (errors, loss, cost).md`. Thuyết minh về sự khác biệt giữa Loss và Cost, vai trò của MSE và Cross-Entropy trong bài toán hồi quy và phân loại. village.
