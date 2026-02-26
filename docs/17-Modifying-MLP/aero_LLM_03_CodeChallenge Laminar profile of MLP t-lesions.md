
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [17 Modifying MLP](../index.md)

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
# Thử thách Lập trình: Hồ sơ Phân tầng của các T-lesions trong MLP (Laminar Profile of MLP T-lesions)

## Tóm tắt (Abstract)
Báo cáo này mở rộng phương pháp can thiệp thống kê từ một tầng duy nhất sang toàn bộ cấu trúc phân tầng (laminar structure) của mô hình BERT Large (24 transformer blocks). Thử thách lập trình tập trung vào việc tối ưu hóa quy trình: chạy các phép thử T-test trực tiếp trong Hook để xác định neurons đặc hiệu, sau đó thực hiện cắt bỏ (lesioning) có hệ thống trên tất cả các tầng. Kết quả tiết lộ hồ sơ tác động (impact profile) của MLP neurons qua các độ sâu của mô hình, cho thấy sự biến thiên phức tạp giữa các tầng và tính chất cục bộ của các biểu diễn ngôn ngữ. Nghiên cứu cũng nhấn mạnh vai trò của trực quan hóa dữ liệu trong việc chẩn đoán các hành vi "kỳ quặc" của mô hình tại các tầng cuối.

---

## 1. Mở Đầu (Introduction)
Hiểu được sự phân bố nhiệm vụ của lớp MLP theo chiều dọc của mô hình là một câu hỏi trung tâm trong mechanistic interpretability. Thử thách này đòi hỏi người học phải nâng cấp quy trình từ "thử nghiệm đơn lẻ" sang "phân tích hệ thống", tự động hóa việc xác định và can thiệp neurons trên quy mô toàn bộ 24 blocks của BERT Large.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Tự động hóa trích xuất đặc trưng (Exercise 1 & 2)
- **Tối ưu hóa:** Thay vì chạy vòng lặp thủ công, chúng ta chuẩn bị trước `target_indices` để Hook có thể truy cập ngay lập tức vào vị trí token "him/her".
- **Dynamic Hook Logic:** Phép thử T-test (kèm hiệu chỉnh đa so sánh FDR) được thực thi ngay trong Forward Pass của dữ liệu nền (baseline dataset).
- **Lưu trữ:** Kết quả được lưu dưới dạng từ điển các vector Boolean (True/False) cho từng tầng, sẵn sàng cho việc can thiệp.

### 2.2. Phân tích Tỷ lệ Neurons đặc hiệu
Tính toán tỷ lệ phần trăm các neurons cho thấy sự khác biệt có ý nghĩa thống kê giữa "him" và "her" tại mỗi tầng. Quan sát này giúp sơ đồ hóa "mật độ tri thức giới tính" theo độ sâu của Transformer.

### 2.3. Thực nghiệm Can thiệp (Exercise 4)
- **Thiết lập:** Sử dụng vòng lặp qua 24 tầng. Tại mỗi tầng, thực hiện vô hiệu hóa (set to 0) các neurons đặc hiệu đã tìm được.
- **Đo lường:** Trích xuất 3 chỉ số:
    1. Tổng độ lớn thay đổi (Global magnitude change).
    2. Tác động cụ thể lên Logit của "her".
    3. Tác động cụ thể lên Logit của "him".

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Hồ sơ Phân tầng (Laminar Profile)
- **Tác động Tổng thể:** Biểu đồ quét cho thấy tất cả các tầng đều đóng góp vào việc hình thành logit cuối cùng, nhưng mức độ đóng góp không đồng đều. Một số tầng "nhạy cảm" hơn với việc mất mát neurons MLP so với các tầng khác.
- **Kết quả Cụ thể:** Việc cắt bỏ thường làm giảm logit của token đúng (giá trị Delta dương), xác nhận vai trò hỗ trợ của neurons MLP cho dự đoán đúng. Tuy nhiên, tại một số tầng (ví dụ tầng 18), chúng ta quan sát thấy hiệu ứng ngược lại hoặc không đáng kể, gợi ý về tính dư thừa (redundancy) hoặc các cơ chế bù trừ trong mạng.

### 3.2. Sự khác biệt Pre-GELU và Post-GELU
- **Lý thuyết:** T-test được chạy trên giá trị Pre-activation (phân phối chuẩn).
- **Thực tế:** Can thiệp được thực hiện trên Post-activation. Việc hiểu rằng các neurons có giá trị âm sẽ bị GELU triệt tiêu giúp giải thích tại sao một can thiệp diện rộng (về mặt thống kê) lại chỉ tạo ra những thay đổi tinh vi về mặt số liệu logit.

---

## 4. Thảo Luận: Những Nuance trong Diễn giải học
Báo cáo chỉ ra rằng việc diễn giải kết quả từ một câu mẫu duy nhất là chưa đủ để đưa ra kết luận tổng quát.
- **Tính bất định:** Các kết quả "quirky" tại các block cuối (như layer 23, 24) có thể do sự tương tác phức tạp với Final Layer Norm hoặc do đặc thù của câu mẫu.
- **Sức mạnh Thống kê:** Cần mở rộng bộ dữ liệu thử nghiệm để tăng Power thống kê, cho phép phân biệt giữa "nhiễu" và "hành vi hệ thống".

---

## 5. Kết Luận
Việc sơ đồ hóa hồ sơ phân tầng của MLP neurons cung cấp cái nhìn sâu sắc về cách tri thức được xử lý theo trình tự. Thử thách này khẳng định rằng sức mạnh của diễn giải học cơ học nằm ở sự kết hợp giữa kỹ thuật lập trình Hook chính xác và tư duy thống kê chặt chẽ. Đây là nền tảng để tiến tới việc loại bỏ các không gian con (subspace removal) trong các nghiên cứu tiếp theo.

---

## Tài liệu tham khảo (Citations)
1. Thử thách Laminar Profile trên BERT Large dựa trên `aero_LLM_03_CodeChallenge Laminar profile of MLP t-lesions.md`. Phân tích hồ sơ tác động của 24 tầng MLP.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Thay thế Trung vị Nối tiếp các Neurons trong Lớp MLP (Successive Median-Replacement of MLP Neurons)](aero_LLM_01_Successive median-replacement of MLP neurons.md) | [Xem bài viết →](aero_LLM_01_Successive median-replacement of MLP neurons.md) |
| [Cắt bỏ Tiệm cận các Neurons MLP trên cơ sở Thống kê (Statistics-based Lesioning of MLP Neurons)](aero_LLM_02_Statistics-based lesioning MLP neurons.md) | [Xem bài viết →](aero_LLM_02_Statistics-based lesioning MLP neurons.md) |
| 📌 **[Thử thách Lập trình: Hồ sơ Phân tầng của các T-lesions trong MLP (Laminar Profile of MLP T-lesions)](aero_LLM_03_CodeChallenge Laminar profile of MLP t-lesions.md)** | [Xem bài viết →](aero_LLM_03_CodeChallenge Laminar profile of MLP t-lesions.md) |
| [Khám phá việc Loại bỏ Không gian con trong MLP (Explorations in Subspace Removal)](aero_LLM_04_Explorations in subspace removal.md) | [Xem bài viết →](aero_LLM_04_Explorations in subspace removal.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
