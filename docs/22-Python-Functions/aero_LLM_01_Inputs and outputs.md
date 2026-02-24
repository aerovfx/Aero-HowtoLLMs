# Nhập môn Python: Hàm, Đầu vào và Đầu ra (Functions, Inputs and Outputs)

## Tóm tắt (Abstract)
Báo cáo này giới thiệu khái niệm về "Hàm" (Functions) trong Python – công cụ cốt lõi để đóng gói và tái sử dụng các khối mã nguồn. Chúng ta nghiên cứu cơ chế vận hành của hàm thông qua quan hệ Đầu vào (Input) và Đầu ra (Output), đồng thời phân tích các hàm dựng sẵn phổ biến như `sum()`, `len()` và `print()`. Nghiên cứu cũng đi sâu vào việc xử lý ngoại lệ khi truyền sai kiểu dữ liệu, cơ chế ẩn đầu ra khi thực hiện phép gán trong Notebook, và thực hiện một thực nghiệm tính toán giá trị trung bình để minh chứng cho nhu cầu sử dụng các thư viện bổ trợ như NumPy.

---

## 1. Khái niệm và Vai trò của Hàm
Trong lập trình, hàm là một tập hợp các dòng mã được thiết kế để thực hiện một tác vụ cụ thể.
- **Tính tái sử dụng:** Thay vì viết lại cùng một thuật toán nhiều lần, lập trình viên đóng gói nó vào một hàm và gọi tên hàm khi cần.
- **Tính cấu trúc:** Hàm giúp chia nhỏ các bài toán phức tạp (như huấn luyện mô hình) thành các module đơn giản, dễ kiểm soát.

---

## 2. Cơ chế Đầu vào và Đầu ra

### 2.1. Tham số Đầu vào (Parameters/Inputs)
Hàm nhận dữ liệu thông qua các dấu ngoặc đơn `()`. 
- **Ví dụ:** Hàm `sum(danh_sách)` nhận một danh sách số và trả về tổng của chúng.
- **Ràng buộc kiểu:** Mỗi hàm yêu cầu loại dữ liệu cụ thể. Việc truyền một chuỗi ký tự (`str`) vào hàm `sum()` sẽ gây ra lỗi `TypeError` vì toán tử cộng (`+`) bị quá tải (overloaded) và không thể xử lý hỗn hợp số và chữ theo cách thông thường.

### 2.2. Giá trị Đầu ra (Return Values/Outputs)
Khi một hàm thực thi xong, nó có thể trả về một kết quả.
- **Gán biến:** Kết quả có thể được lưu trữ vào một biến để sử dụng sau này (ví dụ: `kết_quả = sum(danh_sách)`).
- **Lưu ý về Notebook:** Khi kết quả của hàm được gán cho một biến ở dòng cuối cùng của ô mã, Notebook sẽ không hiển thị giá trị đó ra màn hình. Để xem kết quả, ta cần gọi tên biến đó ở một dòng riêng biệt.

---

## 3. Phân tích Thực nghiệm: Tính Giá trị Trung bình
Qua việc triển khai thuật toán tính trung bình cộng ($Ave18-RAGe = \frac{\sum X}{n}$), chúng ta rút ra được hai quan sát quan trọng:

1. **Độ nhạy Chữ hoa/thường (Case Sensitivity):** Python coi `listCount` và `listcount` là hai thực thể hoàn toàn khác nhau. Một lỗi đánh máy nhỏ trong tên biến sẽ dẫn đến lỗi `NameError`.
2. **Hạn chế của Python Thuần (Base Python):** Python cơ bản không cung cấp sẵn hàm `mean()` hay `ave18-RAGe()`. Để thực hiện các phép toán thống kê này, lập trình viên phải tự xây dựng thuật toán hoặc sử dụng các thư viện chuyên dụng như NumPy.

---

## 4. Tầm quan trọng của các Thư viện (Libraries)
Việc tự viết mọi thuật toán (từ tính trung bình đến các phép toán ma trận phức tạp) là cực kỳ tốn thời gian và dễ sai sót. Đây là lý do tại sao hệ sinh thái Python dựa mạnh vào các thư viện:
- **NumPy:** Xử lý mảng và toán học số học.
- **PyTorch:** Xử lý tensor và học sâu.
- **Pandas:** Quản lý và phân tích dữ liệu bảng.

---

## 5. Kết luận
Hàm là đơn vị cơ bản cấu thành nên logic của mọi ứng dụng AI. Việc nắm vững cách tương tác giữa dữ liệu đầu vào và kết quả đầu ra, cùng với ý thức về các ràng buộc kiểu dữ liệu, là bước đệm thiết yếu để chuyển từ việc viết mã đơn lẻ sang xây dựng các hệ thống tự động hóa phức tạp. Trong các bài học tiếp theo, chúng ta sẽ khám phá cách mở rộng sức mạnh của hàm thông qua việc nhập (import) các thư viện phần mềm chuyên sâu.

---

## Tài liệu tham khảo (Citations)
1. Cơ sở về hàm và tương tác đầu vào/đầu ra trong Python dựa trên `aero_LLM_01_Inputs and outputs.md`. Phân tích hàm `sum()`, `len()` và nhu cầu về thư viện bên thứ ba.
