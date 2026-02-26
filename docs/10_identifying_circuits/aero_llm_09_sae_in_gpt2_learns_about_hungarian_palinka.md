
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
# Huấn Luyện Sparse Autoencoder Trích Xuất Khái Niệm Ngữ Cảnh Palinka Trên GPT-2

## Tóm tắt (Abstract)
Thế giới Toán học mô phỏng dù rất hoàn mỹ nhưng không thể sánh được với sự hỗn loạn của văn bản Ngôn ngữ Tự nhiên. Ở bài thực hành này, ta nâng cấp mô hình Sparse Autoencoder (SAE) để khai thác và trực tiếp nuốt trọn ma trận $MLP\ Activations$ thu được từ mô hình GPT-2 Small, dưới sự kích thích của đoạn văn bản tra cứu Wikipedia về "Palinka" (Một loại Quốc tửu Đông Âu). Bằng việc dùng cơ chế Top-K ép Thưa (Top-K Sparsity) thay vì L1 loss thông thường, ta khảo sát quy trình tìm kiếm Mạch Vi Ngữ (Circuit) đặc thù chịu trách nhiệm cho các ý niệm "Địa lý". Nghiên cứu làm lộ rõ khả năng phân tách Tín hiệu (Denoising / De-mixing) rất mạnh của SAE, song cũng chỉ ra điểm yếu chết người liên quan tới sự nhiễu loạn của "Vung tham số" (Hyperparameter Sensitivity) khi đối diện với Real Datasets quy mô nhỏ. 

---

## 1. Mở Đầu (Introduction)
Dữ liệu sinh ra từ GPT-2 không phải là hàm $Sine$ gọn gàng như bài trước. Một từ đơn như "Romania" sẽ làm cháy sáng hàng trăm Nơ-ron đồng thời, và chúng ta không có một Hệ quy chiếu Tiêu chuẩn (Ground Truth Latent Sources) để chấm điểm đúng sai.
Văn bản đưa vào GPT-2 dài cỡ $220$ Token, bàn về nguồn gốc khu vực địa lý của rượu Palinka (Từ Hungary đến Hy Lạp). Mục tiêu của SAE ở đây là: Bóc tách $768\ MLP\ Neurons$ thành hơn $6000$ Khái niệm Vi Ngữ (Latent Components) trong nút cổ chai phình to, sau đó cố gắng dò tìm xem có bất kỳ Circuit đơn lẻ nào chẻ riêng được khái niệm "Quốc gia / Lãnh thổ" ra khỏi bài văn hay không.

---

## 2. Thiết Lập Thuật Toán Mở Rộng Nhận Diện (Methodology)

### 2.1. Giải Phóng Cấu Trúc Bằng Móc Nối Tied Weights
Với số lượng dữ liệu huấn luyện quá mỏng (chỉ 220 mẫu x $768$ Feature), mà Tầng Latent lại mở rộng lên đến hơn $6000$ Feature, bài toán sẽ lâm vào họa Quá Khớp (Overfitting) vì lượng Tham số Parameter khổng lồ. 
Để vá lỗi, Thuật toán ép dùng kỹ thuật Trọng số Tái chế (Tied Weights):
$\text{Decoder\_Weight} = \text{Encoder\_Weight}^T$
Việc loại bỏ ma trận học Decoder riêng và dùng Ma trận chuyển vị (Transpose) của phần Encoder giúp giảm 50% số lượng biến số tự do, thúc đẩy mô hình học mượt mà hơn.

### 2.2. Kiểm Soát Sự Thưa Thớt Bằng Kỹ Thuật Lọc Top-K Sparsity
Thay vì dùng "thuốc kìm hãm" L1 Regularization truyền thống, ở cấu trúc này hàm $ReLU$ được sử dụng làm Non-linearity kèm Bộ Lọc Chắn (Thresholding).
Thay vì chèn ép mọi thông số, ta chỉ bảo toàn Các kết quả $K$ Điểm sáng mạnh nhất. Ví dụ, cho biến $k = 0.5 \times Dimensions$: Trong mỗi Vector kích hoạt, SAE sẽ dập bỏ $50\%$ số Nơ-ron có giá trị điện áp thấp tuyệt đối về $0$. Kỹ thuật "Diệt cỏ tận gốc" này ép cho Mạng phải chuyển dồn sức chứa vào một hệ sinh thái Siêu thưa thớt (Super Sparsity). 

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Tính Trạng Không Gian Hoạt Hóa Siêu Phân Tán (Sparsity Distribution)
Hậu huấn luyện bằng Adam Optimizer ($100\ Epochs, LR=0.0001$), kết quả Density Matrix (Mật độ tín hiệu) phơi bày cảnh sắc hoang tàn:
- Mức độ Mật độ thưa toàn khối (Sparsity Volume) tàn lụi cực mạnh, khiến tổng Cấu trúc lưới chỉ còn $\sim 6.5\%$ sống sót. 
- Diễn giải đa hình thái: Trong số hơn $6000$ Vi mạch tiềm ẩn (Latent Components): 
  - Phần lớn (Hơn phân nửa) **Tuyệt đối không phản ứng (Tịt ngòi 100%)** trước bất cứ một Token nào trong tổng số 220 Từ vựng đầu vào.
  - Một thiểu số siêu hiếm **Hoạt động liên hồi** cho $100\%$ các Tokens. 
  - Rải rác vài cục bộ có chức năng phản ứng chọn lọc Ngữ pháp.

### 3.2. Rủi Ro Tính Toán Thống Kê Trong Lọc Dữ Liệu
Ta lập một Trọng số Địa lý (Geography Selectivity Score) $=$ Chia giá trị Activations của chuỗi từ $\{}$"Romania", "Hungary", "Greece"$\}$ cho kích hoạt của đám từ thừa thãi như "fruit, alcohol". Rất nhiều Component được điểm Cao ngất ngưởng.
Lỗi diễn giải (Interpretability trap) xảy ra ở đây: 
Việc chèn bộ lọc cơ học dễ khiến ta nhầm tưởng ta đã tìm ra "Bộ Não Địa lý" (Địa danh). Khi vẽ Heatmap, đúng là các Nơ-ron Latent này có "cháy sáng" tại khu vực "Czech Republic, Union". Nhưng đồng thời nó lại nhiễu loạn cháy sáng bùng nổ vô căn cứ ở các cụm từ ngữ pháp vô nghĩa như ngoặc đơn $()$, dấy phẩy $,$, hay chữ "naught". 
Sự thất bại cục bộ trong việc dò tìm ra "Mạch ý nghĩa tuyệt đối" này chứng minh quy luật: Dữ liệu quá nhỏ làm sai lệch Hệ số Thưa thớt (Sparse Representation), tạo ra rác đồng quy (Correlated Redundant Noise) thay vì Trí tuệ trừu tượng sắc nét.

---

## 4. Kết Luận
Autoencoder là thiết chế dò tìm Siêu Khái Niệm cực mạnh, nhưng nó không phải Mũi Tên Bạc thuật giả kim. Khi thao tác trên Datasets thực tế nhưng thiếu khối lượng mẫu, hành động Cưỡng ép Tính chênh lệch (Top-K / Tied Weights) có thể gây ra hiện tượng Phân Mạch Giả (Proxy Circuits) có điểm số Toán học cao nhưng phi ý nghĩa Logic (Semantic Invalidity). Nó nhấn mạnh tiêu chuẩn vàng: Trong Giải diễn Cơ học, Thống Kê Điểm Số bắt buộc phải được đi đôi với Quá Trình Soi Đo Trực Quan (Visual Inspection) một cách chặt chẽ. Ở chương tiếp theo, ta sẽ dùng SAE để quét qua toàn bộ cấu trúc Laminar nhiều tầng cắt thay vì đâm trụ một mỏ đơn lẻ.

---

## Tài Liệu Tham Khảo (Citations)
1. Thí nghiệm ứng dụng Sparse Autoencoder bằng bộ lọc Top-K thay cho L1 Penalty trên dữ liệu GPT-2 Small Hook từ module `aero_LLM_09_SAE in GPT2 learns about Hungarian Palinka.md`. Điểm xuyết thủ thuật Khóa chặn Kiến trúc Transpose Encoder-Decoder Tied Weights.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mạng Mạch Thuật Toán (Circuits) Trong Mô Hình Học Sâu](aero_llm_01_what_is_a_circuit_in_a_dl_model.md) | [Xem bài viết →](aero_llm_01_what_is_a_circuit_in_a_dl_model.md) |
| [Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)](aero_llm_02_isolating_and_investigating_attention_heads.md) | [Xem bài viết →](aero_llm_02_isolating_and_investigating_attention_heads.md) |
| [Thử Thách Lập Trình: Biểu Diễn Phân Bố Nhiệt Laminar Của Trọng Số Chú Ý](aero_llm_03_codechallenge_laminar_profile_of_attention_head_weights.md) | [Xem bài viết →](aero_llm_03_codechallenge_laminar_profile_of_attention_head_weights.md) |
| [Khảo Sát Tương Quan Cụm (Clustering) Vi Mạch (Circuits) Trong Không Gian Giảm Chiều](aero_llm_04_are_circuits_clustered_in_low_dimensional_space.md) | [Xem bài viết →](aero_llm_04_are_circuits_clustered_in_low_dimensional_space.md) |
| [Lý Thuyết Và Ứng Dụng Của Kỹ Thuật Dò Thưa (Sparse Probing)](aero_llm_05_sparse_probing_theory_and_code.md) | [Xem bài viết →](aero_llm_05_sparse_probing_theory_and_code.md) |
| [Thách Thức Của Tín Hiệu Thưa Trong Dữ Liệu Tập Lớn (Statistical Suppression)](aero_llm_06_challenges_with_sparse_logistic_regression_in_large_datasets.md) | [Xem bài viết →](aero_llm_06_challenges_with_sparse_logistic_regression_in_large_datasets.md) |
| [Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI](aero_llm_07_latent_vs_manifest_variables.md) | [Xem bài viết →](aero_llm_07_latent_vs_manifest_variables.md) |
| [Mô Hình Sparse Autoencoders (SAEs): Lý Thuyết Và Kiến Trúc Khôi Phục Vi Mạch Tiềm Ẩn](aero_llm_08_sparse_autoencoders_theory_and_code.md) | [Xem bài viết →](aero_llm_08_sparse_autoencoders_theory_and_code.md) |
| 📌 **[Huấn Luyện Sparse Autoencoder Trích Xuất Khái Niệm Ngữ Cảnh Palinka Trên GPT-2](aero_llm_09_sae_in_gpt2_learns_about_hungarian_palinka.md)** | [Xem bài viết →](aero_llm_09_sae_in_gpt2_learns_about_hungarian_palinka.md) |
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
