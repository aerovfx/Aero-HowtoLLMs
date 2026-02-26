
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
# Khám phá việc Loại bỏ Không gian con trong MLP (Explorations in Subspace Removal)

## Tóm tắt (Abstract)
Báo cáo này trình bày một kỹ thuật can thiệp tân tiến dựa trên đại số tuyến tính để nghiên cứu khối MLP: Loại bỏ không gian con (Subspace Removal). Thay vì triệt tiêu các neurons riêng lẻ, phương pháp này sử dụng Phân tách giá trị suy biến (SVD) để xác định các thành phần chiếm phương sai lớn nhất (principal components) và chiếu chúng ra khỏi dữ liệu hoạt hóa. Thực nghiệm trên GPT-2 Excel (6400 neurons MLP) cho thấy việc loại bỏ chỉ một chiều (1D subspace) mang phương sai lớn nhất có thể phá hủy các cấu trúc hiệp biến (covariance patterns) phân tán trên hàng ngàn neurons, dẫn đến hiệu ứng gợn sóng (ripple effects) tích tụ qua các tầng và làm suy yếu khả năng tận dụng tri thức thế giới của mô hình.

---

## 1. Mở Đầu (Introduction)
Các phương pháp can thiệp trước đây tập trung vào tính thưa (sparsity) của từng neuron. Tuy nhiên, thông tin trong LLM thường được mã hóa phân tán (distributed representation). Báo cáo này đề xuất việc bác bỏ không gian con – một kỹ thuật ngược lại với nén dữ liệu PCA – để kiểm chứng vai trò của các thành phần tiềm ẩn đối với hành vi của mô hình.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Phẫu thuật SVD trên Hoạt hóa MLP
- **Dữ liệu:** Ma trận hoạt hóa $A$ kích thước `[tokens, neurons]`.
- **Phân tách:** $A = U \Sigma V^T$.
- **Can thiệp:** Gán giá trị 0 cho giá trị suy biến lớn nhất ($\sigma_1 = 0$) trong ma trận $\Sigma$.
- **Tái thiết:** Tính toán ma trận đã can thiệp $A_{proj} = U \Sigma_{modified} V^T$.
- **Bù trừ giá trị trung bình:** Một bước quan trọng là cộng lại vector trung bình ($\mu$) của dữ liệu gốc vào $A_{proj}$ để đảm bảo phân phối không bị dịch chuyển quá mức khi đi qua hàm GELU.

### 2.2. Triển khai Hook và Đo lường
Sử dụng mô hình GPT-2 Excel với 48 tầng. Hook được đặt tại tầng `c_fc` để can thiệp vào không gian MLP trước khi phản hồi lại residual stream. Biến quan sát là chuẩn (norm) của sự sai lệch vector hidden states và logit của token dự đoán tiếp theo.

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Sự phá hủy cấu trúc Hiệp biến (Covariance Destruction)
- **Quan sát:** Mặc dù các giá trị neuron cá nhân chỉ thay đổi rất ít sau khi loại bỏ principal component, ma trận hiệp biến bị biến đổi sâu sắc. 
- **Kết luận:** Thành phần chính đầu tiên không chỉ là "nhiễu" mà mang giữ các mô thức tương quan phức tạp giữa hàng ngàn neurons. Loại bỏ nó tương đương với việc xóa bỏ "ngữ cảnh chung" mà các neurons đang cùng chia sẻ.

### 3.2. Hiệu ứng Gợn sóng và Tích tụ (Compounding Effects)
- **Cơ chế:** Thí nghiệm can thiệp tại một tầng duy nhất (ví dụ tầng 24) cho thấy sai lệch trong hidden states tăng dần theo độ sâu của mô hình.
- **Giải thích:** Mỗi block transformer thực hiện một điều chỉnh nhỏ. Nếu đầu vào của block $N+1$ đã bị sai lệch từ block $N$, các điều chỉnh tiếp theo sẽ làm sai lệch đó trầm trọng hơn. Đây là hiện tượng "embedding drift".

### 3.3. Tác động lên Dự đoán Token
- **Phân tích T-test:** Chạy thống kê trên toàn bộ câu cho thấy việc loại bỏ thành phần chính làm giảm logit của token đúng ở hầu hết các tầng.
- **Mối tương quan giữa Phương sai và Tác động:** Có một mối tương quan yếu giữa tỷ lệ phương sai mà thành phần chính chiếm giữ và mức độ sụt giảm hiệu năng. Tuy nhiên, các tầng đầu tiên của mô hình đóng vai trò quyết định, nơi principal component chiếm tỷ trọng rất lớn (đến 37% phương sai).

---

## 4. Thảo Luận: Giới hạn của Giả định PCA
Nghiên cứu chỉ ra rằng mặc dù PCA là công cụ mạnh mẽ, giả định "phương sai lớn nhất tương đương với thông tin quan trọng nhất" không phải lúc nào cũng đúng trong diễn giải học cơ học. Các chiều có phương sai nhỏ hơn đôi khi lại mang thông tin ngữ nghĩa cụ thể hơn. Tuy nhiên, quy trình kỹ thuật được giới thiệu ở đây có thể áp dụng cho bất kỳ phương thức phân tách tuyến tính nào (như ICA hoặc Sparse Autoencoders).

---

## 5. Kết Luận
Báo cáo khẳng định tầm quan trọng của việc nhìn nhận MLP như một không gian vector thay vì chỉ là tập hợp các đơn vị độc lập. Việc bác bỏ không gian con mở ra một hướng đi mới để "tắt" các khái niệm hoặc mô thức tư duy cụ thể trong LLM mà không cần tác động thô bạo lên cấu trúc vật lý của mạng.

---

## Tài liệu tham khảo (Citations)
1. Thực nghiệm Subspace Removal trên GPT-2 Excel dựa trên `aero_LLM_04_Explorations in subspace removal.md`. Phân tích SVD và hiệu ứng tích tụ sai lệch embeddings.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Thay thế Trung vị Nối tiếp các Neurons trong Lớp MLP (Successive Median-Replacement of MLP Neurons)](aero_LLM_01_Successive median-replacement of MLP neurons.md) | [Xem bài viết →](aero_LLM_01_Successive median-replacement of MLP neurons.md) |
| [Cắt bỏ Tiệm cận các Neurons MLP trên cơ sở Thống kê (Statistics-based Lesioning of MLP Neurons)](aero_LLM_02_Statistics-based lesioning MLP neurons.md) | [Xem bài viết →](aero_LLM_02_Statistics-based lesioning MLP neurons.md) |
| [Thử thách Lập trình: Hồ sơ Phân tầng của các T-lesions trong MLP (Laminar Profile of MLP T-lesions)](aero_LLM_03_CodeChallenge Laminar profile of MLP t-lesions.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Laminar profile of MLP t-lesions.md) |
| 📌 **[Khám phá việc Loại bỏ Không gian con trong MLP (Explorations in Subspace Removal)](aero_LLM_04_Explorations in subspace removal.md)** | [Xem bài viết →](aero_LLM_04_Explorations in subspace removal.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
