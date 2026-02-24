# Cực đại hóa Hoạt hóa (Activation Maximization): Cơ sở Lý thuyết và Những thách thức trong LLM

## Tóm tắt (Abstract)
Báo cáo này giới thiệu về "Cực đại hóa Hoạt hóa" (Activation Maximization), một kỹ thuật cốt lõi trong diễn giải học cơ học nhằm xác định đặc điểm mà một neuron cụ thể trong mạng học sâu được điều chỉnh để phản ứng. Thay vì quan sát phản hồi của nơ-ron trước các dữ liệu mẫu có sẵn, phương pháp này sử dụng lan truyền ngược (backpropagation) để tối ưu hóa một nhiễu ngẫu nhiên đầu vào sao cho nó kích hoạt tối đa nơ-ron mục tiêu. Bằng cách đối chiếu các ví dụ thành công từ thị giác máy tính với các đặc thù của ngôn ngữ, báo cáo phân tích bốn giả định cơ bản của phương pháp này và thảo luận về tính khả thi của chúng đối với các Mô hình Ngôn ngữ Lớn (LLM).

---

## 1. Mở Đầu (Introduction)
Hiểu được "ý nghĩa" của một neuron đơn lẻ giữa hàng tỷ đơn vị là thách thức lớn đối với việc diễn giải mô hình. Có hai cách tiếp cận chính:
1. **Quan sát (Observation via Data Sampling):** Đưa một lượng lớn dữ liệu qua mô hình và tìm các mẫu kích hoạt nơ-ron mạnh nhất.
2. **Can thiệp (Intervention via Optimization):** Cố định trọng số mô hình và tinh chỉnh đầu vào để tìm ra "hình ảnh hoặc văn bản lý tưởng" của nơ-ron đó.
Activation Maximization thuộc về cách tiếp cận thứ hai, cho phép ta khám phá không gian biểu diễn mà không bị giới hạn bởi các tập dữ liệu huấn luyện có sẵn.

---

## 2. Quy trình Tối ưu hóa ngược (Reverse Optimization)
- **Normal Training:** Dữ liệu ($X$) cố định, trọng số mô hình ($\theta$) thay đổi để giảm thiểu tổn thất (Loss).
- **Activation Maximization:** Trọng số mô hình ($\theta$) cố định, dữ liệu đầu vào ($X$) thay đổi thông qua Gradient Descent để cực đại hóa hoạt hóa của nơ-ron đích $a_{i}$.
Kết quả cuối cùng là một "bản đồ đặc trưng" phản ánh chân thực nhất thiên kiến nội tại của nơ-ron.

---

## 3. Các Giả định Cốt lõi của Phương pháp
Việc áp dụng Activation Maximization dựa trên bốn giả định quan trọng:

### 3.1. Giả định về Đặc trưng Đơn lẻ (Unit Feature Representation)
Giả định rằng mỗi nơ-ron đại diện cho một khái niệm con người có thể hiểu được (ví dụ: nơ-ron "mắt chó" trong CNN hoặc nơ-ron "thì quá khứ" trong LLM). Tuy nhiên, trong thực tế, các nơ-ron thường tham gia vào các "biểu diễn phân tán" (polysemanticity), khiến việc cô lập một ý nghĩa duy nhất trở nên khó khăn.

### 3.2. Giả định về Tầm quan trọng của Hoạt hóa (Activation as Importance)
Giả định rằng cường độ hoạt hóa tỷ lệ thuận với tầm quan trọng của thông tin. Tuy nhiên, trong sinh học và đôi khi trong AI, sự ức chế hoạt hóa hoặc sự phối hợp giữa một cụm nơ-ron mới mang lại thông tin chính xác.

### 3.3. Tính Liên tục và Khả vi của Không gian Đầu vào
Phương pháp yêu cầu không gian tối ưu hóa phải "mượt" (smooth). Điều này đúng với hình ảnh (pixel mang giá trị liên tục), nhưng rất thách thức với ngôn ngữ vốn mang tính rời rạc (Discrete). Không có trạng thái "nằm giữa" 1/3 từ "táo" và "chuối".

### 3.4. Tính Có thể Diễn giải (Human Interpretability)
Giả định rằng kết quả tối ưu hóa phải có ý nghĩa với logic của con người. Nghiên cứu chỉ ra rằng nhiều nơ-ron nhìn thế giới theo cách "kỳ dị" (rubbish images) mà mắt người không thể nhận diện được, dù chúng vẫn hoạt động chính xác trong kiến trúc của mô hình.

---

## 4. Kết Luận
Activation Maximization là một công cụ mạnh mẽ để "buộc" mô hình tiết lộ các cấu trúc ẩn. Dù các giả định về tính liên tục và tính đơn ngữ (monosemanticity) thường bị vi phạm trong LLM, phương pháp này vẫn cung cấp những hiểu biết quan trọng về cách mô hình mã hóa thế giới vượt ra ngoài các tập dữ liệu mẫu. Những bài thực hành tiếp theo sẽ tập trung vào việc triển khai kỹ thuật này bằng PyTorch và Gradient Descent.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở lý thuyết về Activation Maximization dựa trên `aero_LLM_01_Activation maximization via gradient ascent (theory).md`. Phân tích các ví dụ từ CNN và các rào cản khi áp dụng lên ngôn ngữ.
