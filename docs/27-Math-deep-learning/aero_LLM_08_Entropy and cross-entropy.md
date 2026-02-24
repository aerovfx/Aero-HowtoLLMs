# Toán học trong Học sâu: Entropy và Cross-Entropy (Entropy)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu về lý thuyết thông tin trong học sâu, tập trung vào hai khái niệm cốt lõi: Entropy và Cross-Entropy. chúng ta phân tích Entropy Shannon như một thước đo của sự "bất ngờ" (surprise) hoặc độ bất định trong một hệ thống dữ liệu. Nghiên cứu đi sâu vào cơ chế của Cross-Entropy trong việc đo lường khoảng cách giữa phân phối xác suất thực tế (labels) và phân phối dự đoán của mô hình (predictions). Bằng các thực nghiệm trên NumPy và PyTorch, chúng ta minh chứng cách thức biến đổi các bài toán phân loại thành các bài toán tối ưu hóa thông qua hàm mất mát Binary Cross Entropy (BCE), đồng thời làm rõ các yêu cầu kỹ thuật về định dạng tensor và thứ tự tham biến trong lập trình thực tiễn.

---

## 1. Entropy Shannon: Thước đo Độ bất định

Trong lý thuyết thông tin, Entropy không đại diện cho sự hỗn loạn vật lý mà đại diện cho lượng thông tin hoặc độ khó dự đoán của một biến ngẫu nhiên.
- **Nguyên lý cực đại:** Entropy đạt giá trị cao nhất khi xác suất các sự kiện là tương đương nhau (ví dụ $p=0.5$ trong tung đồng xu), vì khi đó chúng ta hoàn toàn không biết kết quả nào sẽ xảy ra.
- **Nguyên lý cực tiểu:** Khi một sự kiện trở nên chắc chắn ($p=0$ hoặc $p=1$), sự bất ngờ biến mất và Entropy tiến về 0.
- **Công thức:** $H(x) = -\sum p(x) \log p(x)$. Dấu âm giúp đảm bảo giá trị Entropy luôn dương vì logarit của xác suất (từ 0 đến 1) luôn âm.

---

## 2. Cross-Entropy trong Huấn luyện Mô hình

Cross-Entropy là công cụ để so sánh hai phân phối xác suất khác nhau:
- **Phân phối thực tế ($p$):** Thường là các nhãn (labels) dạng "one-hot" (ví dụ: [1, 0] cho mèo).
- **Phân phối dự đoán ($q$):** Là đầu ra của hàm Softmax từ mô hình (ví dụ: [0.9, 0.1]).
- **Mục tiêu tối ưu:** Cực tiểu hóa Cross-Entropy đồng nghĩa với việc đẩy dự đoán của mô hình ($q$) tiến sát về phía sự thật khách quan ($p$). Khi mô hình dự đoán chính xác tuyệt đối, Cross-Entropy sẽ đạt giá trị tối thiểu.

---

## 3. Binary Cross Entropy (BCE) và Sự đơn giản hóa

Đối với các bài toán phân loại nhị phân (có/không, mèo/chó), công thức Cross-Entropy được đơn giản hóa thành:
$$BCE = -[p \log(q) + (1-p) \log(1-q)]$$
Trong thực tế học sâu, vì $p$ thường chỉ bằng 0 hoặc 1, công thức này lại càng đơn giản hơn: nó chỉ đơn thuần là giá trị âm logarit của xác suất mà mô hình gán cho lớp đúng. Nếu mô hình càng tự tin vào lớp đúng, giá trị mất mát (loss) càng nhỏ.

---

## 4. Thực thi Kỹ thuật trên PyTorch

Việc sử dụng PyTorch yêu cầu sự chính xác cao về cú pháp:
- **Hàm `F.binary_cross_entropy`:** Yêu cầu tham số đầu tiên là dự đoán từ mô hình và tham số thứ hai là nhãn thực tế. Việc đảo ngược thứ tự này sẽ dẫn đến kết quả sai lệch nghiêm trọng.
- **Quản lý Tensor:** PyTorch không chấp nhận danh sách Python (`list`) thông thường cho các phép toán này. Dữ liệu phải được chuyển đổi thành `torch.Tensor` trước khi tính toán.
- **Tính ổn định số học:** PyTorch thường tích hợp sẵn các kỹ thuật xử lý để tránh lỗi khi $\log(0)$ (giá trị không xác định), giúp quá trình huấn luyện diễn ra trơn tru ngay cả khi mô hình đưa ra dự đoán cực đoan.

---

## 5. Kết luận
Entropy và Cross-Entropy là "ngôn ngữ" để đo lường sự thông minh của một mô hình. Một mô hình càng học tốt thì Cross-Entropy giữa dự đoán của nó và thực tế càng thấp. Thấu hiểu các khái niệm này giúp nhà nghiên cứu không chỉ nắm vững cơ chế của các hàm mất mát (loss functions) mà còn có cái nhìn sâu sắc về cách thức mà thông tin được luân chuyển và định lượng bên trong các kiến trúc LLM hiện đại.

---

## Tài liệu tham khảo (Citations)
1. Lý thuyết thông tin Shannon và Cross-Entropy trong học sâu dựa trên `aero_LL_08_Entropy and cross-entropy.md`. Phân tích độ bất định, công thức BCE và thực thi hàm mất mát trong PyTorch.
