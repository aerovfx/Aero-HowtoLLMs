
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [28 Gradient descent](../index.md)

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
# Học sâu: Thực thi Hạ giang trong Không gian 1 Chiều (1D Gradient Descent)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về quy trình triển khai thực tế của thuật toán Hạ giang (Gradient Descent) trong không gian một chiều bằng ngôn ngữ lập trình Python và thư viện NumPy. chúng ta phân tích sự tương tác giữa hai siêu tham số (hyperparameters) cốt lõi: tốc độ học (learning rate) và số lượng vòng lặp huấn luyện (epochs). Nghiên cứu thực hiện các thực nghiệm mô phỏng để minh chứng cách thức mô hình tự điều chỉnh từ một điểm khởi tạo ngẫu nhiên tiến về điểm tối ưu $x=0.5$, đồng thời cảnh báo về các rủi ro khi thiết đặt tham số không phù hợp dẫn đến hiện tượng bùng nổ hoặc hội tụ chậm của mô hình.

---

## 1. Cấu trúc Thực thi và Khởi tạo

Để trực quan hóa Gradient Descent, chúng ta định nghĩa một hàm mất mát đa thức và đạo hàm tương ứng của nó. Quá trình bắt đầu bằng việc chọn một vị trí ngẫu nhiên trên trục $x$:
- **Khởi tạo ngẫu nhiên:** Một giá trị $x$ được chọn từ tập dữ liệu ban đầu làm "điểm bắt đầu" cho hành trình tìm kiếm cực tiểu.
- **Vòng lặp (Epochs):** Mô hình thực hiện lặp đi lặp lại việc tính đạo hàm tại vị trí hiện tại và dịch chuyển ngược hướng với đạo hàm đó. Mỗi vòng lặp hoàn chỉnh được gọi là một Epoch.

---

## 2. Vai trò của Tốc độ học (Learning Rate)

Tốc độ học ($\eta$) quyết định "độ dài" của mỗi bước đi trong quá trình hạ giang:
- **Tốc độ học quá lớn (ví dụ $\eta = 1$):** Mô hình sẽ "nhảy" quá đà, tạo ra các giá trị tham số lớn khủng khiếp (theo ký hiệu khoa học $E+70$). Điều này làm hỏng hoàn toàn quá trình huấn luyện và khiến thuật toán không thể hội tụ.
- **Tốc độ học quá nhỏ (ví dụ $\eta = 0.001$):** Mô hình đi những bước cực kỳ thận trọng. Sau 100 epoch, nó vẫn chưa thể tiếp cận được điểm tối ưu $0.5$, đòi hỏi chi phí tính toán lớn hơn (nhiều epoch hơn) để đạt được kết quả mong muốn.
- **Tầm quan trọng:** Việc "tinh chỉnh" (tuning) tốc độ học là kỹ năng quan trọng nhất của một kỹ sư AI.

---

## 3. Mối liên hệ giữa Epochs và Hội tụ

Nghiên cứu chỉ ra rằng số lượng vòng lặp và tốc độ học có mối quan hệ tỷ lệ nghịch trong việc đạt được mục tiêu:
- **Sự bù đắp:** Nếu chúng ta giảm tốc độ học đi 10 lần, chúng ta thường cần tăng số lượng epoch lên gấp 10 lần (ví dụ từ 100 lên 1000) để mô hình có đủ thời gian "lăn" tới đáy thung lũng.
- **Tiệm cận:** Biểu đồ thực nghiệm cho thấy giá trị tham số và giá trị đạo hàm tiến gần đến mục tiêu theo dạng đường cong tiệm cận. Khi càng gần điểm tối ưu, tốc độ thay đổi càng chậm lại vì giá trị đạo hàm lúc này rất nhỏ, dẫn đến các bước cập nhật trở nên tinh vi hơn.

---

## 4. Chẩn đoán mô hình qua Trực quan hóa

Thông qua việc lưu trữ tham số tại mỗi epoch, chúng ta có thể vẽ được "hành trình học tập" của mô hình:
- **Trục Tọa độ:** Cho thấy cách tham số $x$ thay đổi từ vị trí khởi tạo (có thể âm hoặc dương) và dần ổn định tại giá trị $0.5$.
- **Trục Đạo hàm:** Minh chứng mục tiêu của thuật toán là đưa đạo hàm về 0. Nếu đường biểu diễn đạo hàm vẫn còn dốc ở epoch cuối cùng, điều đó có nghĩa là mô hình cần được huấn luyện thêm.

---

## 5. Kết luận
Thực thi Gradient Descent trong không gian 1D là bài tập "vỡ lòng" nhưng chứa đựng toàn bộ bản chất của học sâu hiện đại. Qua thực nghiệm, chúng ta nhận thấy rằng thành công của một mô hình AI không chỉ nằm ở thuật toán mà còn ở sự phối hợp nhịp nhàng giữa tốc độ học và thời gian huấn luyện. Việc thấu hiểu các dynamics (động lực học) này trong không gian đơn giản là tiền đề vững chắc để nhà nghiên cứu làm việc với các hệ thống LLM phức tạp, nơi các tham số được tính bằng hàng tỷ đơn vị.

---

## Tài liệu tham khảo (Citations)
1. Quy trình thực thi thủ công và phân tích tham số học tập dựa trên `aero_LL_03_Gradient descent in 1D.md`. Phân tích thực nghiệm trên NumPy, vai trò của Epochs vs Learning Rate và chẩn đoán hội tụ tiệm cận.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
