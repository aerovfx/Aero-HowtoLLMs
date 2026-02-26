
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
# Khảo Sát Tương Quan Cụm (Clustering) Vi Mạch (Circuits) Trong Không Gian Giảm Chiều

## Tóm tắt (Abstract)
Nghiên cứu này trình bày một thí nghiệm dùng phương pháp gom cụm thống kê (Clustering analysis) để dò tìm Mạch Vi Ngữ (Circuits) trong tầng đa lớp (MLP Layers) của GPT-2 Medium. Mặc dù chúng ta thành công trong việc chỉ ra các Tế bào Nơ-ron (Neurons) có chức năng phân cực mạnh mẽ khi xử lý danh xưng (Ví dụ: "Him" vs "Her") thông qua phép kiểm định Student's t-test, việc cố gắng nhúng (Embedding) và gọt giũa tập Tế bào này trên không gian 2D bằng thuật toán t-SNE, DBSCAN, và K-means đã thất bại trong việc tạo ra một cấu trúc Mạng Mạch cụm (Clustering structure) khả dụng. Tuy nhiên, sự thất bại này là bài học quý giá về sức mạnh và giới hạn của các phương pháp phân rã hình học tự động trên Dữ liệu hoạt hóa lớn.

---

## 1. Mở Đầu (Introduction)
Trong Cơ học hệ phức tạp (Complex systems) hoặc Khoa học thần kinh, một "Mạng mạch" (Circuit) thường được định vị bởi một cụm các tế bào thần kinh hoạt động đồng pha (Correlated activations). Trực giác Toán học cũng đề xuất phương pháp tương tự trên Mô hình ngôn ngữ lớn (LLM): Nếu ta lọc được tất cả các Neurons có xu hướng kích hoạt phản ứng trước đại từ nhân xưng, sau đó phân tách không gian của chúng, ta hẳn sẽ tìm được các Đảo cụm độc lập (Islands or Clusters) tạo nên Mạch Vi Ngữ. Thí nghiệm thực tiễn dưới đây chỉ ra điểm đứt gãy giữa lý thuyết Sinh lý và số học của Không gian Vector.

---

## 2. Tiết Thiết Lập (Methodology)

### 2.1. Kiểm Định t-test Độc Lập Mở Rộng
Toàn bộ Tế bào tương quan `MLP` tại Block số 5 của GPT-2 Medium (Layer 5) được trích xuất. Tầng MLP này có kích thước nở phình lên $4$ lần thành chiều dài Vector = $4096$ chiều.
Ta tiêm (Hook) tập huấn luyện siêu vi (54 câu có đại từ "him", và 54 câu y hệt nhưng tráo bằng "her"). Với từng Neural 1 trong chuỗi 4096 Neurons, ta đo lường độ phản ứng trung bình giữa hai phiên bản ngữ cảnh và chạy bài toán kiểm định giả thuyết `t-test`.
Khoảng $30\%$ lượng Neurons rơi vào ngưỡng Ý nghĩa thống kê (Statistical significance), sau khi áp dụng Hình phạt hiệu chỉnh P-value Bonferroni: $P_{threshold} = \frac{0.05}{4096}$. Tập hợp 1523 nơ-ron chiến thắng này được gọi là "Hạt giống Tương tác Hình-thái" (Morphological responsive candidates).

### 2.2. Phương Pháp Chéo Không Gian Thấp Chiều (Lower Dimensional Embeddings)
Thay vì làm việc với số lượng chiều quá lớn, 1523 nơ-ron này được chích qua ba phễu giảm chiều / phân cụm:
1. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Ép trục tạo ra không gian 2 chiều (2D projection).
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Tự động phát hiện đốm tập trung thông qua bán kính khuếch tán.

$$
3. **K-Means Clustering:** Bắt ép chia cắt các điểm lân cận thành K = 13 Lõi Cấu trúc (Centroids).
$$

---

## 3. Khảo Sát & Giải Phẫu Mô Hình (Analysis)

### 3.1. Sự Tách Rời Tuyệt Định của Thuật Toán t-SNE
Quy chiếu t-SNE đem đến một sự phân cực đối xứng đồ thị cực kỳ hoàn mỹ giữa Tập tế bào "Dương tính" (ủng hộ "Him") và "Âm tính" (ủng hộ "Her"). 
Tuy nhiên, điều lạ lùng là nó **không tạo ra một cụm tiểu đảo nào**. Đám mây phân phối mượt mà không cho thấy bất cứ một Tiểu Mạng Mạch (Subnetworks) cục bộ hay sự ngắt quãng (Discretization) nào.

### 3.2. Cạm Bẫy Ảo Giác Trong Noise Clusters
Việc DBSCAN cố gắng khoanh vùng tạo ra các kết quả hỗn loạn. Chỉ cần xê dịch yếu tố độ phân tán nhiễu Parameter (Epsilon/Perplexity values) đôi chút, toàn thể kiến trúc Cụm của Mạng thay đổi trầm trọng. Các nhóm cụm K-means nhặt vừa Tế bào âm, vừa Tế bào dương vào cùng một lớp nhãn chỉ đơn thuần vì sự sát nhập vị trí (Local Proximity). 
Ma trận tương quan độ đo Cosine (Cosine similarity matrix) - thước đo chuẩn nhất để soi độ "kết băng" của Neurons - cũng trắng xóa và sạch nhẵn, không thấy các khối Block Đồng thuận (Consensus Blocks) trên đường chéo chính. 
*Hệ quả:* Hệ thống các Tế bào Nhạy Cảm Ngữ Pháp này chạy dọc độc lập thay vì móc ngoặc tuyến tính với nhau tạo thành một Circuit "khép kín" dễ nhận biết bằng hình học Euclidean.

---

## 4. Kết Luận
Bài toán "Giải phẫu cụm t-SNE/K-means" trên Lớp giãn nở (MLP Expansion) là ví dụ kinh điển cho Lỗi Cố Định Cơ Thể (Biological fixed-form fallacy) trong AI Deep Learning. Băng nhóm Neurons phục vụ tác vụ Phân tách Giới tính hoàn toàn không tổ chức định cư dưới dạng các Cụm vật lý quy tụ (Spatial groupings). 
"Không có thử nghiệm nào là vô ích" (Alexander Graham Bell). Bằng cách loại bỏ lối mòn khai thác khoảng cách không gian, ta chính thức mở đường đi đến những công cụ vĩ mô bẻ gãy hệ phương trình phi tuyến tính (Như Phương pháp Eigen Phân rã dạng ma trận Generalized Eigendecomposition - GED), để tìm kiến trúc tiềm ẩn dạng "Quang phổ" thay vì "Trọng điểm".

---

## Tài liên tham khảo (Citations)
1. Thí nghiệm đo lường T-tests và mô phỏng K-means từ `aero_LLM_04_Are circuits clustered in low-dimensional space.md`. Ứng dụng hàm API `sklearn.manifold.TSNE` và `sklearn.cluster.DBSCAN`.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Mạng Mạch Thuật Toán (Circuits) Trong Mô Hình Học Sâu](aero_llm_01_what_is_a_circuit_in_a_dl_model.md) | [Xem bài viết →](aero_llm_01_what_is_a_circuit_in_a_dl_model.md) |
| [Cô Lập Và Thăm Dò Khối Chú Ý (Attention Heads)](aero_llm_02_isolating_and_investigating_attention_heads.md) | [Xem bài viết →](aero_llm_02_isolating_and_investigating_attention_heads.md) |
| [Thử Thách Lập Trình: Biểu Diễn Phân Bố Nhiệt Laminar Của Trọng Số Chú Ý](aero_llm_03_codechallenge_laminar_profile_of_attention_head_weights.md) | [Xem bài viết →](aero_llm_03_codechallenge_laminar_profile_of_attention_head_weights.md) |
| 📌 **[Khảo Sát Tương Quan Cụm (Clustering) Vi Mạch (Circuits) Trong Không Gian Giảm Chiều](aero_llm_04_are_circuits_clustered_in_low_dimensional_space.md)** | [Xem bài viết →](aero_llm_04_are_circuits_clustered_in_low_dimensional_space.md) |
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
