# Thử thách Lập trình: Điều chỉnh Phủ định trong Nơ-ron MLP (Phần 3)

## Tóm tắt (Abstract)
Báo cáo này tổng kết thử thách nghiên cứu về cơ chế phủ định trong GPT-2 Large bằng cách phân tích sự biến đổi của nơ-ron xuyên suốt toàn bộ chiều sâu của mạng Transformer. Chúng ta thực hiện một phân tích hệ thống trên 36 tầng, định lượng mật độ và hiệu năng của các "nơ-ron phủ định". Kết quả thực nghiệm cho thấy một xu hướng suy giảm rõ rệt về cả số lượng nơ-ron chuyên biệt hóa lẫn độ chính xác phân loại khi đi về phía các tầng cuối. Khám phá này ủng hộ giả thuyết về sự chuyển dịch chức năng của mô hình: từ việc giải mã đặc tính của token hiện tại sang việc tích hợp ngữ cảnh để dự báo tương lai.

---

## 1. Phân tích chức năng mở rộng (Exercise 5)

### 1.1. Sự chọn lọc ngữ nghĩa (Semantic Selectivity)
Thông qua bản đồ nhiệt hoạt hóa, chúng ta quan sát thấy nơ-ron có Beta cao nhất không chỉ phản ứng với các từ phủ định thuần túy (*not, neither, won't*) mà còn kích hoạt mạnh với các từ mang sắc thái tiêu cực hoặc suy tàn (ví dụ: *rusty, dead, corroded*). 
- **Giải thích:** Điều này chứng tỏ nơ-ron đã học được một "tính năng" (feature) trừu tượng hơn là chỉ một danh sách từ vựng. Nó mã hóa một khái niệm logic/ngữ nghĩa rộng hơn về sự phủ định hoặc sự thiếu vắng.

---

## 2. Xu hướng xuyên Tầng (Exercise 6)

### 2.1. Mật độ Nơ-ron có Ý nghĩa Thống kê
Chúng ta chạy hồi quy trên toàn bộ 36 tầng và áp dụng hiệu chỉnh Bonferroni khắt khe ($p < 0.05 / 5120$).
- **Tầng đầu (Early Layers):** Lên đến 70% số nơ-ron trong tầng bộc lộ tính chọn lọc phủ định rõ rệt.
- **Tầng cuối (Late Layers):** Tỷ lệ này giảm xuống chỉ còn khoảng 25%. Mặc dù giảm mạnh, nhưng vẫn còn khoảng 1000 nơ-ron duy trì được tín hiệu, cho thấy cơ chế này vẫn được bảo tồn một phần ở giai đoạn cuối.

### 2.2. Hiệu năng Dự báo (Accuracy)
Độ chính xác trung bình của các nơ-ron được chọn lọc (Significant Positive Beta) cũng bộc lộ xu hướng tương tự:
- Đạt đỉnh khoảng 75-80% tại các tầng thấp.
- Giảm xuống mức 65% ở các tầng cao nhất (vượt trên mức ngẫu nhiên 50%).

---

## 3. Kiến giải về Kiến trúc Transformer

### 3.1. Chuyển dịch từ Hiện tại sang Tương lai
Sự suy giảm của các "nơ-ron phủ định" ở các tầng sâu có thể được giải thích bằng nhiệm vụ chính của mô hình: **Dự báo Token tiếp theo (Next Token Prediction)**.
- **Giai đoạn đầu:** Residual stream chứa thông tin đậm đặc về thuộc tính của chính token đó. Các nơ-ron MLP tập trung giải mã logic của token hiện tại.
- **Giai đoạn cuối:** Mô hình ưu tiên việc chuẩn bị Logits cho từ tiếp theo. Do đó, các biểu diễn về "hiện tại" (như phủ định) bị mờ nhạt dần để nhường chỗ cho các dự báo về "tương lai".

---

## 4. Kỹ thuật Lập trình: Ma trận Mặt nạ (Masked Arrays)
Để tính toán trung bình trên các tập hợp nơ-ron thỏa mãn đồng thời hai điều kiện ($\beta > 0$ và $p < \alpha$), chúng ta sử dụng `np.ma.masked_array`.
- **Lưu ý quan trọng:** Trong NumPy, mask có giá trị `True` nghĩa là điểm dữ liệu bị che khuất (không tính). Vì vậy, để lấy các nơ-ron ý nghĩa, chúng ta phải sử dụng toán tử nghịch đảo (`~`) trên mask điều kiện.

---

## 5. Kết Luận Chung
Thử thách này làm nổi bật tính phức tạp và thú vị của Mechanistic Interpretability. Chúng ta đã biến một câu hỏi ngôn ngữ học trừu tượng ("Mô hình xử lý sự phủ định như thế nào?") thành một bài toán định lượng với dữ liệu thực nghiệm. Việc hiểu rõ bản chất thống kê và xu hướng của nơ-ron theo tầng là bước chuẩn bị quan trọng để khám phá các thành phần khác như các đầu Attention (Attention Heads) trong các nghiên cứu tiếp theo.

---

## Tài liệu tham khảo (Citations)
1. Tổng kết Negation tuning trên GPT-2 Large dựa trên `aero_LLM_22_CodeChallenge Negation tuning in MLP neurons (part 3).md`. Phân tích xu hướng Accuracy theo tầng và giả thuyết chuyển dịch chức năng.
