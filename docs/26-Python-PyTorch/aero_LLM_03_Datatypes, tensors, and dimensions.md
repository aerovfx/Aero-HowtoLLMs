
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [26 Python PyTorch](../index.md)

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
# Nhập môn PyTorch: Kiểu dữ liệu, Tensor và Kích thước (Datatypes, Tensors, and Dimensions)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về cấu trúc dữ liệu cốt lõi của thư viện PyTorch: **Tensor**. Chúng ta phân tích sự tương đồng và khác biệt giữa Tensor và các khái niệm đại số tuyến tính truyền thống như vô hướng (scalar), vectơ (vector) và ma trận (matrix). Nghiên cứu đi sâu vào cách thức PyTorch biểu diễn các mảng đa chiều thông qua hệ thống dấu ngoặc lồng nhau, cơ chế xác định hình dạng (shape), và kỹ thuật quản lý các chiều dữ liệu (dimensions). Đây là kiến thức nền tảng để hiểu cách các mô hình ngôn ngữ lớn (LLM) lưu trữ và xử lý các tập hợp tham số khổng lồ dưới dạng các khối dữ liệu đa chiều.

---

## 1. Hệ sinh thái PyTorch và các Module chính

PyTorch là thư viện hàng đầu trong nghiên cứu học sâu nhờ tính linh hoạt và cú pháp trực quan.
- **`import torch`:** Nạp thư viện chính.
- **`torch.nn` (viết tắt là `nn`):** Chứa các lớp (classes) để xây dựng kiến trúc mạng nơ-ron.
- **`torch.nn.functional` (viết tắt là `F`):** Cung cấp các hàm số độc lập như hàm kích hoạt (activation functions) hoặc hàm mất mát (loss functions).

---

## 2. Phân cấp Tọa độ trong Tensor

Trong đại số tuyến tính và máy học, Tensor là một thuật ngữ tổng quát cho các mảng có thứ tự bất kỳ:
- **0D Tensor (Scalar - Số vô hướng):** Một điểm dữ liệu đơn nhất.
- **1D Tensor (Vector - Vectơ):** Một dãy số liên tiếp (dạng hàng hoặc cột).
- **2D Tensor (Matrix - Ma trận):** Một bảng dữ liệu có hàng và cột (tương tự bảng tính Excel).
- **3D Tensor và cao hơn:** Một khối dữ liệu (cube) hoặc siêu khối (hypercube). Ví dụ: một video có thể coi là 3D tensor (chiều cao x chiều rộng x thời gian).

---

## 3. Bản chất của Hình dạng (Shape) và Kích thước

Cách thức PyTorch xác định số chiều của một Tensor phụ thuộc vào mức độ lồng nhau của các dấu ngoặc vuông `[]`:
- **Đặc trưng bậc thấp:** Một Tensor chứa `1.0` (vô hướng) có shape rỗng. Nếu bao bọc bởi `[1.0]`, nó trở thành vectơ có kích thước `[1]`. Nếu là `[[1.0]]`, nó trở thành ma trận `1x1` (shape `[1, 1]`).
- **Tensor đa chiều:** Ví dụ, một Tensor có shape `[2, 2, 3]` đại diện cho một khối gồm 2 "tờ" ma trận, mỗi tờ có 2 hàng và 3 cột.

---

## 4. Thực nghiệm Chỉ mục và Cắt lát (Indexing)

Việc truy cập dữ liệu trong Tensor phụ thuộc vào số chiều mà nó sở hữu:
- **Cắt lát 2D:** Truy cập một chỉ mục trong ma trận `2x3` sẽ trả về một vectơ (một hàng).
- **Cắt lát 3D:** Truy cập một chỉ mục trong Tensor `2x2x3` sẽ trả về một ma trận nguyên bản nằm ở vị trí đó trong khối.
Nhà nghiên cứu cần tư duy về dữ liệu như các lát cắt của một khối đa chiều để có thể trích xuất các đặc trưng (features) một cách chính xác.

---

## 5. Kết luận
Tensor không chỉ là một mảng số học mà là một thực thể vận tải thông tin trong các mạng nơ-ron. Việc phân biệt rõ các cấp độ từ vô hướng đến đa chiều, cũng như thấu hiểu cơ chế "đóng gói" dữ liệu qua các dấu ngoặc, là điều kiện tiên quyết để vận hành các thuật toán học sâu. Trong nghiên cứu LLM, khả năng quản lý kích thước tensor giúp chúng ta điều phối luồng dữ liệu giữa các lớp chú ý (attention layers) và các lớp kết nối đầy đủ (fully connected layers) một cách hiệu quả và khoa học.

---

## Tài liệu tham khảo (Citations)
1. Cấu trúc Tensor và quản lý kích thước trong PyTorch dựa trên `aero_LL_03_Datatypes, tensors, and dimensions.md`. Phân tích phân cấp từ Scalar đến Tensor đa chiều và kỹ thuật cắt lát dữ liệu.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
