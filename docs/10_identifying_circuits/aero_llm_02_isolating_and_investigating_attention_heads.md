
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [10 identifying circuits](index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../index.md)
- [📚 Module 01: LLM Course](../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)

## Tóm tắt (Abstract)
Báo cáo trình bày phương pháp giải phẫu một trong những linh hồn cốt lõi của kiến trúc Transformer: Cơ chế Đa chú ý (Multi-head Attention). Bằng việc theo dõi cách biến đổi và phân mảnh ma trận Tương tác $Q, K, V$ dọc theo hệ chiều nhúng thành các module đầu độc lập (Heads), ta có thể nhận thức được sự chuyên biệt hóa luồng thông tin của LLM. Một phát hiện đáng chú ý là sự thiên vị điểm âm (Negative shift) của các Tích vô hướng gốc (Raw Attention Scores), điều này giải thích toán học cho cơ chế "Sparsity" - triệt tiêu sự nhiễu loạn từ token không liên quan. Báo cáo cũng đề xuất phương pháp mô hình hóa mật độ hạt nhân Kernel Density Estimation (KDE) thay cho Scatter plots nội suy, giúp trực quan hóa phân bổ xác suất một cách khoa học.

---

## 1. Mở Đầu (Introduction)
Trong Mạng Mạch của Deep Learning, việc quy tụ hàng khối Head lại với nhau thông qua ma trận trộn tuyến tính $W_o$ (Linear mix matrix) là chìa khóa tổng hợp kiến thức ngôn ngữ. Tuy nhiên, nếu chúng ta có thể chẻ nhỏ và truy cập vào từng "Não bộ phụ" (Head) riêng lẻ đang phân tích gì, ta sẽ hiểu được cơ chế hoạt động vi mô (Mechanistic Interpretability). Công việc này đòi hỏi kết hợp phương trình Attention cốt lõi: $Softmax(\frac{QK^T}{\sqrt{d_k}})V$ kết hợp thao tác ma trận tinh vi.

---

## 2. Tiết Thiết Lập (Methodology)

### 2.1. Nhắc Lại Thuật Toán Attention và Mặt Nạ Causal Label (Masking)
$Q$ (Query) đại diện cho "Token hiện tại đang tìm kiếm gì?", còn $K$ (Keys) đại diện cho "Các token cũ giữ thông tin gì đáng giá?". Tích vô hướng $QK^T$ đo lường sự tương thích. 
Tuy nhiên, Transformer là bộ dự báo chuỗi theo thời gian (Autoregressive), nó bắt buộc không được "Nhìn trộm" tương lai. Lớp Mặt Nạ $M$ (Masking matrix) được phủ lên $QK^T$: Các tọa độ ở tam giác dưới (Quá khứ) nhận mức $1$, tọa độ ở tam giác trên (Tương lai) nhận $-$\infty$$. Khi qua hàm phi tuyến kích hoạt $Softmax$, hàm số $e^{-$\infty$}$ biến mất thành điểm $0$ tuyệt đối. 
*Hệ quả dị biệt:* Mảnh Token đầu tiên của chuỗi không có quá khứ, nên toàn bộ thông số liên kết ngược bị xóa sổ $\to$ tự gán $100\%$ lực chú ý vào chính bản thân nó (Outlier error).

### 2.2. Trích Xuất Attention Đầu Phụ (Heads Isolation)
Trên GPT-2 Small, ma trận sau khi hook lấy cắp từ `hook_h.attn.c_attn` sẽ là một khối $768 \times 2304$. Do $2304 = 768 \times 3$, nó đang gộp chung tệp $Q, K, V$. 
1. Cắt lấy $1/3$ đầu tiên ta được Ma trận thuần Query $Q$ (Kích thước: $SequenceLength \times 768$).
2. Tiếp tục dùng hàm `torch.split` chiếu dọc theo chiều Dimensions (768), văm thành $12$ khúc. Kết quả: $12$ Attention Heads, mỗi Head thu được ma trận $SequenceLength \times 64$.
3. Tại điểm này, ta tính Tích vô hướng (Dot products) nội bộ cho từng Head riêng biệt để lấy Raw Attention Scores.

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Sự Thiên Vị Âm Tính Vô Hướng (Negative Raw Attention Shift)
Theo lý thuyết xác suất, khi lấy mẫu ma trận điểm nhân với nhau, phân bổ đồ thị phân tán (Scatter plots) của $QK^T$ (Raw attention scores) thường nên nằm ở dạng đối xứng ngay quanh mốc zero. Tuy nhiên, GPT-2 điều hướng trọng số lệch mạnh mẽ về khu vực cực âm (Negative numbers). 
Đây không phải là lỗi. Nó là một thủ thuật Tối ưu Thưa (Sparsity mechanism). Khi số gốc mang giá trị âm sâu, hàm kích hoạt $Softmax$ sẽ dập toàn bộ tập xác suất này xấp xỉ mức $0$. Việc LLM đẩy hầu hết điểm tương tác xuống mức âm giúp triệt tiêu hoàn toàn các mối quan hệ Token dư thừa từ quá khứ (suppression), qua đó để nhường chỗ, vinh danh cho một số rất nhỏ các kết nối ngữ pháp thực sự ý nghĩa (Ví dụ: tính từ liên kết danh từ).

### 3.2. Nội Suy Phân Bổ Mật Độ KDE (Kernel Density Estimation)
Phương thức biểu diễn bằng các chấm phân tán Scatter plots trở nên vô dụng nếu dữ liệu lớn cồng kềnh qua hàng chục Layers. Phương thức thay thế: KDE (Mô hình hóa mật độ hạt nhân).
KDE coi một điểm phân tán là một tâm thu hút phân phối vi mô (Gaussian blur). Bằng cách convolve lặp và cộng dồn toàn bộ các màng sương Gaussian có độ băng thông nhất định (Bandwidth parameter), ta biến các số thô (Discrete values) thành đường cong phổ phân bổ mượt mà (Probability distribution curve). 

---

## 4. Kết Luận
Việc tách lẻ các Head giải phẫu quá trình tính tương phản Query-Key đưa lại lời giải đáp vì sao $Softmax$ có năng lực xử lý ngôn ngữ sạch sẽ và sắc bén: Nhờ mô hình tự động "Dìm" phổ Tích vô hướng gốc về các chỉ số siêu nhỏ để loại bỏ nhiễu. Phương pháp tách chẻ ma trận trực tiếp và áp dụng hệ tính toán mật độ hạt nhân (KDE) là bậc thang dữ liệu hoàn hảo trước khi đi sâu vẽ dải viền (Laminar Profiles) Attention head, bước cơ bản để khám phá "Mạng mạch" ở mô-đun kế tiếp bài thử thách.

---

## Tài Liệu Tham Khảo (Citations)
1. Cơ chế cắt mảnh ma trận và phân chia Tensor trong `aero_LLM_02_Isolating and investigating attention heads.md`. Thí nghiệm vẽ KDE thông qua thư viện `scipy.stats.gaussian_kde` và minh họa dịch chuyển âm Tích vô hướng.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mạng Mạch Thuật Toán (Circuits) Trong Mô Hình Học Sâu](aero_llm_01_what_is_a_circuit_in_a_dl_model.md) | [Xem bài viết →](aero_llm_01_what_is_a_circuit_in_a_dl_model.md) |
| 📌 **[Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)](aero_llm_02_isolating_and_investigating_attention_heads.md)** | [Xem bài viết →](aero_llm_02_isolating_and_investigating_attention_heads.md) |
| [Thử Thách Lập Trình: Biểu Diễn Phân Bố Nhiệt Laminar Của Trọng Số Chú Ý](aero_llm_03_codechallenge_laminar_profile_of_attention_head_weights.md) | [Xem bài viết →](aero_llm_03_codechallenge_laminar_profile_of_attention_head_weights.md) |
| [Khảo Sát Tương Quan Cụm (Clustering) Vi Mạch (Circuits) Trong Không Gian Giảm Chiều](aero_llm_04_are_circuits_clustered_in_low_dimensional_space.md) | [Xem bài viết →](aero_llm_04_are_circuits_clustered_in_low_dimensional_space.md) |
| [Lý Thuyết Và Ứng Dụng Của Kỹ Thuật Dò Thưa (Sparse Probing)](aero_llm_05_sparse_probing_theory_and_code.md) | [Xem bài viết →](aero_llm_05_sparse_probing_theory_and_code.md) |
| [Thách Thức Của Tín Hiệu Thưa Trong Dữ Liệu Tập Lớn (Statistical Suppression)](aero_llm_06_challenges_with_sparse_logistic_regression_in_large_datasets.md) | [Xem bài viết →](aero_llm_06_challenges_with_sparse_logistic_regression_in_large_datasets.md) |
| [Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI](aero_llm_07_latent_vs_manifest_variables.md) | [Xem bài viết →](aero_llm_07_latent_vs_manifest_variables.md) |
| [Mô Hình Sparse Autoencoders (SAEs): Lý Thuyết Và Kiến Trúc Khôi Phục Vi Mạch Tiềm Ẩn](aero_llm_08_sparse_autoencoders_theory_and_code.md) | [Xem bài viết →](aero_llm_08_sparse_autoencoders_theory_and_code.md) |
| [Huấn Luyện Sparse Autoencoder Trích Xuất Khái Niệm Ngữ Cảnh Palinka Trên GPT-2](aero_llm_09_sae_in_gpt2_learns_about_hungarian_palinka.md) | [Xem bài viết →](aero_llm_09_sae_in_gpt2_learns_about_hungarian_palinka.md) |
| [Khảo Sát Phân Tầng Kích Hoạt (Laminar Profile) Qua Sparse Autoencoder](aero_llm_10_codechallenge_laminar_profile_of_autoencoder_sparsity.md) | [Xem bài viết →](aero_llm_10_codechallenge_laminar_profile_of_autoencoder_sparsity.md) |
| [Nhận Diện Khái Niệm Xuyên Tâm Với Phân Rã Giá Trị Riêng Suy Rộng (Generalized Eigendecomposition - GED)](aero_llm_11_non_orthogonal_latent_components_via_eigendecomposition_theory_and_demo_.md) | [Xem bài viết →](aero_llm_11_non_orthogonal_latent_components_via_eigendecomposition_theory_and_demo_.md) |
| [Rạch Ròi Giới Tính (Him vs Her) Bằng Generalized Eigendecomposition Trong MLP](aero_llm_12_generalized_eigendecomposition_separates_him_from_her_in_mlp.md) | [Xem bài viết →](aero_llm_12_generalized_eigendecomposition_separates_him_from_her_in_mlp.md) |
| [Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 1)](aero_llm_13_codechallenge_ged_for_category_isolation_across_layers_part_1_.md) | [Xem bài viết →](aero_llm_13_codechallenge_ged_for_category_isolation_across_layers_part_1_.md) |
| [Thử Thách Lập Trình (Code Challenge): Tách Nhóm GED Đa Tầng (Phần 2) & Kiểm Chứng Chéo](aero_llm_14_codechallenge_ged_for_category_isolation_across_layers_part_2_.md) | [Xem bài viết →](aero_llm_14_codechallenge_ged_for_category_isolation_across_layers_part_2_.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
