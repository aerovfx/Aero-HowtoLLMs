
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [10 Identifying circuits](../index.md)

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
# Khảo Sát Phân Tầng Kích Hoạt (Laminar Profile) Qua Sparse Autoencoder

## Tóm tắt (Abstract)
Thử thách lập trình (Code Challenge) này hướng đến việc thu thập Đới Kích Hoạt (Laminar Profile) của toàn bộ các điểm nút MLP chạy dọc theo từ lớp (Layer) đầu tới cuối của Hệ Sinh thái GPT-2 (Cả bản Small $12\ Layers$ và Large $36\ Layers$). Mục tiêu không nhắm vào việc giải cấu trúc hay định hướng ý nghĩa một Feature Latent đơn lẻ, mà nhằm vẽ ra Bước sóng Mật Độ Khởi động (Activation Density) dưới tư cách là những quần thể vi mô. Bằng cách nối ống hút Activations trực tiếp từ bài thi Wikipedia về Công nghệ đèn LED, nghiên cứu chứng minh được một Quy luật tịnh tiến mạnh mẽ: Sự Thưa thớt bị triệt tiêu dần, và Mật độ Kích hoạt (Density) tăng vọt một cách tuyến tính khi ta đi sâu vào các Tầng đáy (Deeper Layers).

---

## 1. Mở Đầu (Introduction)
Thay vì chìm đắm trong việc "Đọc tâm trí" một Latent Component như đã làm với khái niệm "Geography", báo cáo này bay lên góc nhìn Vĩ mô (Macro-scale). Câu hỏi đặt ra là: Autoencoder sẽ phản ứng khác nhau thế nào khi nén và giải nén dữ liệu Điện áp từ các lớp Transformer khác nhau?
Dữ liệu đào tạo được đổi khẩu vị bằng cách không copy-paste truyền thống, mà là sử dụng Thư viện `requests` để thâu tóm toàn bộ Source Code HTML $\sim 500,000$ ký tự tương đương $52,000$ Tokens của trang Wikipedia "LEDs". Quá trình cạo văn bản (Text scraping) tuân thủ tiêu chuẩn: Lọc rác Header/Footers bằng cách khoanh vùng từ cờ `<mw-body-content>` đến `id="references"`.

---

## 2. Tiết Thiết Lập Ghi Nhận Xuyên Tâm (Methodology)

### 2.1. Kiến Trúc Thu Thập Ma Trận Đa Tầng (Multi-layer Hooks)
Ta thiết lập một vòng lặp FOR khổng lồ quét qua toàn bộ Layer $L \in \{0 \dots n\}$.
1. Thu nhận Dữ liệu thô (Input): Chọn ngẫu nhiên $10,240$ Tokens $\to$ Định dạng thành khối Tensor Kích thước `[10, 1024]`.
2. Trích xuất Activations: Tại mỗi Layer $L$, móc Hook thu thập giá trị xuất ra từ cổng $MLP$ nội tại. Dữ liệu mảng Đầu tiên (Zero-th Token) thường chứa các hiện tượng khởi bào hỗn loạn cực đoan (Unusual Outliers) do cơ chế Context Loading, nên bắt buộc bị cắt bỏ (Slicing out token $[0]$).

### 2.2. Huấn Luyện Cục Bộ (Per-layer SAE Training)
Tại duy nhất mỗi Tầng $L$:
- SAE được khởi tạo Mới Hoàn Toàn (Khôi phục ma trận Trọng số về Random).
- Training loop chạy với $75\ Epochs$, sử dụng hàm L1 Sparsity và MSE Loss.
- Thu thập lại giá trị Density cuối cùng (Mức độ chiếm đóng của Tầng Latent $> 0$) và Giá trị Gốc MSE (Khả năng khôi phục).

---

## 3. Khảo Sát & Phác Họa Hành Vi (Analysis)

### 3.1. Sự Tăng Trưởng Tuyến Tính Của Mật Độ Kích Hoạt (Density Profiling)
Quan sát biểu đồ Line Graph trung bình Density cho thấy một quỹ đạo tịnh tiến mạnh mẽ từ $Layer\ 0 \to Layer\ 12$ (Đối với GPT2-Small) và vươn tới $Layer\ 36$ (Đối với GPT2-Large).
Các lớp Mở Đầu (Early layers), với năng lực tính toán còn gần với Từ Vựng (Shallow Embeddings), ghi nhận khả năng thu gọn rất xuất sắc của SAE: Hầu hết Activation bị nén về $0$. Tuy nhiên, khi luồng văn bản chảy vào các Lớp Cuối (Deeper layers), Mô Hình đối diện với Trọng trách Cốt Lõi: Tiên đoán Token tiếp theo (Next-Token Prediction). Ở giai đoạn này, hàm nghĩa của văn bản đã trở nên Tích chập đa chiều (Incorporating broad context). Do đó SAE bất lực trong việc duy trì trạng thái Thưa thớt (Sparsity); Các Vi mạch phải bung nở hỏa lực đồng loạt để gánh vác các logic nội suy đa hợp tuyến. Sự chênh lệch Mật độ này đúng trên cả $2$ Size mô hình, cho thấy đây là Bản tính Hàm Tính Toán (Algorithmic Nature) chứ không phải Do Hạn chế Tham số.

### 3.2. Vùng "Đệm" Dễ Thở Nhất (Sweet Spot of Reconstruction)
Song song đó, Đồ thị Độ trượt Khôi Phục (Final Loss) bộc lộ hiện tượng Đáy chảo (U-shape) thoai thoải: Layer Đầu tiên và Lớp Cuối cùng luôn mắc sai số tái hồi cao nhất. Kỳ lạ thay, các Lớp Chuyển Mạch ở Trung Tâm (Ví dụ Layer $5-7$ của Small, hoặc $10-20$ của GPT2-Large) ghi nhận MSE Loss chạm đáy tối thiểu. Điều này ngụ ý Không gian Biểu diễn (Representation Space) ở giữa mô hình là tĩnh tại, bớt bị dồn nén hay vỡ rạc nhất, giúp Autoencoder "Dịch thuật" tín hiệu dễ dàng hơn.

---

## 4. Kết Luận
Việc chỉ mổ xẻ Một Tầng Mạng để phán xét là thiển cận, bởi Cấu trúc Kích hoạt bên trong LLM vận động và biến thiên liên tục theo Lát cắt Laminar. Thí nghiệm quét tầng bằng SAE trên GPT-2 củng cố góc nhìn Vĩ Mô: Sự thưa thớt (Sparsity) - thứ quyết định Tính Diễn Giải (Interpretability) - giảm dần tỉ lệ nghịch với Độ Sâu của Mạng. Càng tiệm cận tới Tầng Final Output, mọi cơ chế nén gọn sẽ bị đánh sập để nhường chỗ cho cụm Logic phân tán, dập tắt hi vọng chia để trị (Divide-and-conquer) ở giai đoạn xuất xưởng. Ở báo cáo kế tiếp, ta sẽ chia tay với hệ thống Gradient của Autoencoders để làm quen với Năng lượng Phân rã Trị Riêng (Genearlized Eigendecomposition).

---

## Tài liên tham khảo (Citations)
1. Thí nghiệm khảo sát Vĩ mô Sparse Autoencoder chạy ngang Laminar Profile của GPT-2, tham vấn tài liệu `aero_LLM_10_CodeChallenge Laminar profile of autoencoder sparsity.md`. Ghi nhận sự khác biệt của cơ chế Scraping HTTP Body (`mw-body-content`) bằng `requests`.
<!-- Aero-Footer-Start -->
---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
