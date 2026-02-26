
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
# Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 2) & Kiểm Chứng Chéo

## Tóm tắt (Abstract)
Tiếp nối cơ sở Hạ tầng GED đã xây dựng trong Phần 1, phần 2 này kích hoạt vòng lặp `FOR` chạy dọc toàn bộ chiều sâu của Mô hình Language Model. Báo cáo này tổng hợp quá trình diễn dịch đồ thị thông kê T-test, hệ số Tương quan Pearson (Correlation) và Max Eigenvalues qua các tầng. Chúng ta phát hiện ra những quy luật thú vị: Mức độ tương quan (Correlation) giữa "Pattern Him" và "Pattern Her" cao ở các lớp Đầu, nhưng trượt dốc về $0$ ở các lớp Cuối. Cùng với đó, khả năng Phân Ly Dữ Liệu (Separability Ratio) đạt đỉnh tại khoảng $1/3$ thân mạng. Cuối cùng, việc Kiểm định mù (Blind Test) trên Fineweb mang lại kết quả cực kỳ ý nghĩa (Significant T-values) – bảo chứng rằng Bộ Định Tuyến (Vectors) do GED tìm thấy không phải là cú Overfitting may mắn, mà nó thực sự thấu hiểu Khái niệm Giới tính.

---

## 1. Mở Đầu (Introduction)
Sau khi có được hai Hòm dụng cụ: `dim_red(layer)` để ép dữ liệu xuống Không gian con PCA, và Hàm `run_ged(train_data, pca_evecs)` để nhào nặn Không Gian Tối Ưu, ta cần khảo sát Sinh lý học của mô hình GPT-2:
1. Độ sâu của Transformer Block (từ $Layer\ 1 \to Layer\ 12$) thay đổi cách Biểu Diễn Giới tính như thế nào?
2. Có hiện tượng Học Vẹt (Memorization / Overfitting) hay không?

---

## 2. Tiết Thiết Lập Đánh Giá Đa Tầng (Methodology)

### 2.1. Ma Trận Đựng Kết Quả Đa Chiều (Result Tensor)
Bên bài thí nghiệm, ta định hình một Ma trận PyTorch `[12_Layers, 7_Analyses, 2_Stats]`:
- Chiều thứ nhất: Số thứ tự Lớp (Layer $0 \to 11$).
- Chiều thứ hai: Các kiểu Testing (T-test Him vs Her, R-Pearson Correlation Pattern, Max Eigenvalue...)
- Chiều thứ ba: Lưu thông số Độ lớn $T_{value}$ (Magnitude) và Giá trị chuẩn xác suất $p\text{-value}$ (Bonferroni Corrected).
Trong Vòng lặp:
**Bước Lọc Mù (Out-of-sample Evaluation):** Toàn bộ Data Trích lấy từ Fineweb (Test-set) **TUYỆT ĐỐI** KHÔNG ĐƯỢC CHẠY LẠI THUẬT TOÁN PCA và GED. Chúng chỉ đơn thuần đứng im và bị Phóng (Projected) xuyên qua các lưới Lọc `PCA_Eigenvectors` và `GED_Eigenvectors` tạo ra bởi Bộ Từ Vựng Nhân tạo, sau đó mới dùng Phép thử T-test để đo độ Cắt Xẻ (Separation).

### 2.2. Kiểm định T-Test Hai Phía (Bi-directional T-Testing)
Vì GED tự sinh ra 2 Lớp Cột (2 Top Eigenvectors, 1 dùng S=Him_R=Her, 1 dùng S=Her_R=Him). Chúng ta thi hành T-test chéo ngược: Đưa Khối dữ liệu 'Her' Đi qua Màng Lọc 'Him', lúc này Lượng Activations bị tiêu biến dần dẫn tới hiệu số Trung bình Phương sai cực nhỏ (Magnitude T-Value Âm). Do thuật toán GED giải Vi Phân dẫn đến **Bất định Dấu (Sign Indeterminacy)**, ta giải tỏa chuyện lằng nhằng của Dấu + / Dấu - bằng việc đặt Tuyệt Đối $|T_{value}|$.

---

## 3. Khảo Sát & Phác Họa Hành Vi (Analysis)

### 3.1. Sự Sụp Đổ Tương Quan Cơ Tính (Correlation Plummeting)
Khi Vẽ Trục $x=Layer$, $y=Pearson\ R\ (|Correlation|)$ giữa Khối Vector Hướng Pattern HIM và Khối Hướng HER. 
- Tại $4$ Layers đầu: $R$ rất cao. Phản ánh đúng Thực tại: "Him" và "Her" vốn cùng mang một Hệ đặc tính cú pháp (Grammar function) giống y hệt nhau làm Đại từ Nhân xưng (Pronouns). Sự khác biệt vật lý của mạng trong lúc vừa nhai Nuốt Token (Shallow layers) là RẤT ÍT. 
- Tại Các Layers Cuối (Deeper layers): $R \to 0$. Transformer đã chuyển trạng thái từ việc Phân Tích Cú pháp Nội Tại $\to$ Tiến tới Tiên Đoán Tương Lai (Next-Token Prediction). Lúc này, Hành vi, Logic, Cấu trúc không gian của con Đực và con Cái rẽ nhánh hoàn toàn, khiến các Hàm Pattern bay ra hai phương trời riêng biệt.

### 3.2. Hiệu Năng Vượt Rào Chống Overfit (Significant Out-of-Sample Performance)
Đồ thị biểu diễn Test Data (Chấm Tròn và Dấu $x$ Đỏ): Hầu như toàn Cầu (12/12 Layers) đều ghi nhận Mức độ Tách Bạch Khác Biệt Giới Tính trên FineWeb Test Set là Cực Kì Đáng Tin Cậy ($p < 0.05 / 12$). Dù dữ liệu Fineweb cực kỳ nhiễu loạn về Ngữ cảnh, Vectơ Mạch Giới Tính (Gender Circuit Vectors) do GED cọ xát ra vô cùng Kiên Cố. Sợi dây Cấu trúc Giới tính đã thực sự bị Cắt Ra và Cô Lập được đúng định tuyến.

### 3.3. Bí Ẩn Về Her Separability (Khả năng Tách biệt Của Her)
Biểu đồ *Tiền Số Giá trị riêng (Eigenvalue Ratio $E_1/E_2$)* - Đo lường Độ Sắc bén Của Phép Cắt. Đồ thị Vồng lên tạo Đỉnh đồi chóp tại Khoảng Layer số 4. Cực kì kỳ lạ, Năng lực Cắt "HER" tách khỏi "HIM" luôn Rõ nét và bén vót hơn việc phải Cắt "HIM" khỏi "HER". Hiện tượng này xuất hiện trên cả GPT-2 Small và Phiên bản Khổng Khồ GPT-2 XL, chứng tỏ nó là một Feature Hàm ẩn Thuộc Về Cấu Trúc Khối Dữ Liệu Internet (Có thể do Tần suất xuất hiện, ngữ pháp phân cực của Phái nữ trong Data thô nổi bật hơn) - Một hiện tượng mà Tác giả không thể Lý giải tận cùng.

---

## 4. Kết Luận
Bài thực hành Phân rã Toán Học suy rộng GED chốt lại Bức Tranh Toàn Cảnh về Khả năng Bóc Mạch Cơ học trong LLMs: Bằng cách Kết Hợp Đại số Tuyến Tính Giải Tích ($\mathbf{R}^{-1}\mathbf{S}$ Eigenvalue) dưới sự bảo vệ của Không gian Nhỏ Đầu Tiên (PCA Pre-filter), ta hoàn toàn đủ sức Khoét Vách những Khái niệm Cực kì Vô Hình (Giới tính Đại từ) ra khỏi bề mặt Hoạt Hóa khổng lồ. Và quan trọng nhất, Hiện Tượng Tích Trụ Đặc Điểm và Phân Ly (Separability / Decoupling) diễn ra mạnh mẽ nhất Lớp Giữa - Lòng chảo biến thiên Logic đích thực của một Large Language Model.

---

## Tài Liệu Tham Khảo (Citations)
1. Thực nghiệm Blind-Test GED Validation trên nền FineWeb Text, dựa trên `aero_LLM_14_CodeChallenge GED for category isolation across layers (part 2).md`. Khẳng định hiện tượng sụp đổ Pattern Correlation ở Lát cắt Output Layers trong kiến trúc mô hình Transformer.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mạng Mạch Thuật Toán (Circuits) Trong Mô Hình Học Sâu](aero_LLM_01_What is a circuit in a DL model.md) | [Xem bài viết →](aero_LLM_01_What is a circuit in a DL model.md) |
| [Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)](aero_LLM_02_Isolating and investigating attention heads.md) | [Xem bài viết →](aero_LLM_02_Isolating and investigating attention heads.md) |
| [Thử Thách Lập Trình: Biểu Diễn Phân Bố Nhiệt Laminar Của Trọng Số Chú Ý](aero_LLM_03_CodeChallenge Laminar profile of attention head weights.md) | [Xem bài viết →](aero_LLM_03_CodeChallenge Laminar profile of attention head weights.md) |
| [Khảo Sát Tương Quan Cụm (Clustering) Vi Mạch (Circuits) Trong Không Gian Giảm Chiều](aero_LLM_04_Are circuits clustered in low-dimensional space.md) | [Xem bài viết →](aero_LLM_04_Are circuits clustered in low-dimensional space.md) |
| [Lý Thuyết Và Ứng Dụng Của Kỹ Thuật Dò Thưa (Sparse Probing)](aero_LLM_05_Sparse probing theory and code.md) | [Xem bài viết →](aero_LLM_05_Sparse probing theory and code.md) |
| [Thách Thức Của Tín Hiệu Thưa Trong Dữ Liệu Tập Lớn (Statistical Suppression)](aero_LLM_06_Challenges with sparse logistic regression in large datasets.md) | [Xem bài viết →](aero_LLM_06_Challenges with sparse logistic regression in large datasets.md) |
| [Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI](aero_LLM_07_Latent vs. manifest variables.md) | [Xem bài viết →](aero_LLM_07_Latent vs. manifest variables.md) |
| [Mô Hình Sparse Autoencoders (SAEs): Lý Thuyết Và Kiến Trúc Khôi Phục Vi Mạch Tiềm Ẩn](aero_LLM_08_Sparse autoencoders theory and code.md) | [Xem bài viết →](aero_LLM_08_Sparse autoencoders theory and code.md) |
| [Huấn Luyện Sparse Autoencoder Trích Xuất Khái Niệm Ngữ Cảnh Palinka Trên GPT-2](aero_LLM_09_SAE in GPT2 learns about Hungarian Palinka.md) | [Xem bài viết →](aero_LLM_09_SAE in GPT2 learns about Hungarian Palinka.md) |
| [Khảo Sát Phân Tầng Kích Hoạt (Laminar Profile) Qua Sparse Autoencoder](aero_LLM_10_CodeChallenge Laminar profile of autoencoder sparsity.md) | [Xem bài viết →](aero_LLM_10_CodeChallenge Laminar profile of autoencoder sparsity.md) |
| [Nhận Diện Khái Niệm Xuyên Tâm Với Phân Rã Giá Trị Riêng Suy Rộng (Generalized Eigendecomposition - GED)](aero_LLM_11_Non-orthogonal latent components via eigendecomposition (theory and demo).md) | [Xem bài viết →](aero_LLM_11_Non-orthogonal latent components via eigendecomposition (theory and demo).md) |
| [Rạch Ròi Giới Tính (Him vs Her) Bằng Generalized Eigendecomposition Trong MLP](aero_LLM_12_Generalized eigendecomposition separates him from her in MLP.md) | [Xem bài viết →](aero_LLM_12_Generalized eigendecomposition separates him from her in MLP.md) |
| [Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 1)](aero_LLM_13_CodeChallenge GED for category isolation across layers (part 1).md) | [Xem bài viết →](aero_LLM_13_CodeChallenge GED for category isolation across layers (part 1).md) |
| 📌 **[Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 2) & Kiểm Chứng Chéo](aero_LLM_14_CodeChallenge GED for category isolation across layers (part 2).md)** | [Xem bài viết →](aero_LLM_14_CodeChallenge GED for category isolation across layers (part 2).md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
