# Nhập môn Python: Kỹ thuật Trực quan hóa Dữ liệu với Matplotlib (Plotting Dots and Lines)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về thư viện `matplotlib`, bộ công cụ tiêu chuẩn trong Python để chuyển đổi dữ liệu số thành các biểu đồ trực quan. Chúng ta phân tích cơ chế hoạt động của module `pyplot`, từ việc vẽ các điểm tọa độ đơn lẻ đến việc xây dựng các đường cong phức tạp thông qua tập hợp các đoạn thẳng biên độ nhỏ. Nghiên cứu cũng đi sâu vào các kỹ thuật tinh chỉnh đồ họa như tùy biến dấu mốc (markers), màu sắc (colors), nhãn dán (labels) và chú giải (legends). Đây là kiến thức nền tảng để nhà nghiên cứu phân tích xu hướng hội tụ của hàm mất mát (loss function) và phân phối activations trong các mạng nơ-ron.

---

## 1. Cơ chế Cơ bản của Matplotlib Pyplot

### 1.1. Quy ước Nhập thư viện
Thư viện thường được nạp dưới tên viết tắt phổ quát: `import matplotlib.pyplot as plt`. Việc sử dụng tiền tố `plt` giúp mã nguồn trở nên tinh gọn khi thực hiện nhiều lệnh vẽ biểu đồ liên tiếp.

### 1.2. Biểu diễn Điểm (Markers)
Hàm `plt.plot(x, y, 'marker')` yêu cầu tọa độ X và Y.
- **Ký hiệu dấu mốc:** `'o'` (hình tròn), `'s'` (hình vuông), `'p'` (ngũ giác), `'^'` (tam giác).
- **Ký hiệu màu sắc:** `'r'` (đỏ), `'b'` (xanh dương), `'k'` (đen), `'m'` (tím sen).
- **Kích thước:** Tham số `markerSize` cho phép điều chỉnh độ lớn của dấu mốc để tăng khả năng quan sát trên các màn hình có độ phân giải khác nhau.

---

## 2. Quản lý Chú giải và Hiển thị (Legends & Display)

### 2.1. Kỹ thuật Gán nhãn (Labeling)
Khi vẽ nhiều tập dữ liệu trên cùng một trục tọa độ, việc sử dụng tham số `label='tên_dữ_liệu'` trong mỗi hàm `plot` là cần thiết để phân biệt các luồng thông tin.

### 2.2. Kích hoạt Chú giải (Legend Activation)
Thông tin nhãn sẽ không hiển thị cho đến khi hàm `plt.legend()` được gọi. Hàm này tự động tổng hợp các nhãn đã khai báo và đặt chúng vào vị trí tối ưu trên biểu đồ.

### 2.3. Trình diễn kết quả (`plt.show()`)
Hàm `plt.show()` dọn dẹp các thông tin địa chỉ bộ nhớ dư thừa và chỉ hiển thị biểu đồ cuối cùng. Trong môi trường như Google Colab, biểu đồ có thể tự động xuất hiện, nhưng việc sử dụng `plt.show()` là thói quen tốt để đảm bảo mã nguồn tương thích với mọi môi trường lập trình (IDE).

---

## 3. Bản chất của Đường cong trong Máy tính
Trong đồ họa máy tính, không có "đường cong" tuyệt đối. Mọi đường cong (như hàm Sine) thực chất là sự kết nối của hàng trăm hoặc hàng nghìn đoạn thẳng cực nhỏ.
- **Độ phân giải (Resolution):** Khi số lượng điểm tọa độ đủ lớn (ví dụ: 1001 điểm), mắt người sẽ cảm nhận đó là một đường cong mượt mà.
- **Tùy biến đường kẻ:** Sử dụng `'-'` cho nét liền, `'--'` cho nét đứt và `':'` cho nét chấm.

---

## 4. Thực nghiệm Toán học: Vẽ sóng Sine
Bằng cách kết hợp NumPy (`np.linspace` cho trục X và `np.sin` cho giá trị Y), nhà nghiên cứu có thể mô phỏng các tín hiệu sóng một cách chính xác. Việc lồng ghép các biểu thức toán học trực tiếp vào hàm vẽ (`plt.plot(a, b**2)`) minh chứng tính linh hoạt cao của Matplotlib trong các tác vụ tính toán khoa học.

---

## 5. Kết luận
Matplotlib không chỉ là một công cụ vẽ hình đơn thuần mà là một ngôn ngữ giao tiếp giữa dữ liệu và con người. Việc làm chủ các kỹ thuật từ vẽ điểm cơ bản đến các đường cong phức tạp cung cấp cho nhà nghiên cứu khả năng "nhìn thấy" dữ liệu bên trong các ma trận trọng số khổng lồ, từ đó đưa ra các quyết định điều chỉnh mô hình chính xác và hiệu quả.

---

## Tài liệu tham khảo (Citations)
1. Thao tác vẽ điểm và đường với Matplotlib trong Python dựa trên `aero_LL_01_Plotting dots and lines.md`. Phân tích dấu mốc, màu sắc, chú giải và bản chất đường cong trong đồ họa máy tính.
