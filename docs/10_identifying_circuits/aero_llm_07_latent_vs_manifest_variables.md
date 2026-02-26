
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
# Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI

## Tóm tắt (Abstract)
Báo cáo này làm rõ phương pháp luận thống kê áp dụng trong ngành Khoa học Hệ thống phức tạp, cụ thể là Mechanistic Interpretability cho LLM. Để mô phỏng và định lượng các khái niệm trừu tượng bên trong Không gian Nơ-ron (Như "Sự chú ý", "Lừa dối", hay "Ảo giác"), ta không thể dùng thước đo vật lý hay số liệu hiển ngôn (Manifest variables) để ghi nhận trực tiếp. Thay vào đó, chúng bắt buộc phải được quy đổi thành các Cấu trúc hàm tiềm ẩn (Latent Constructs). Bài viết phân tách giới hạn của phương trình Manifest và mở đầu cho sự cấp thiết của các mô hình Hồi quy trung gian như Sparse Autoencoders (SAE) hay Generalized Eigen Decomposition (GED) trong việc nội suy hành vi đa chiều.

---

## 1. Mở Đầu (Introduction)
Trong Khoa học nhận thức và Thống kê học phân tích, có một lằn ranh rõ rệt giữa hai thể chế dữ liệu:
1. **Biến Hiển Ngôn (Manifest Variables/Observable Data):** Là những đại lượng vật lý có thể đếm, đo đạc hoặc tính toán tuyệt đối thông qua thiết bị máy móc hoặc hàm số. Ví dụ: Chiều cao $(cm)$, Lương tháng $(VND)$, Giá trị phần trăm (Logits output), hay Số hệ số phân phối Kích hoạt điện áp của lớp MLP.
2. **Biến Tiềm Ẩn (Latent Variables/Constructs):** Là các khái niệm/học thuyết mà tư duy con người đồng thuận sự tồn tại của nó, tỷ lệ tuyến tính/phi tuyến với các đại lượng vật lý, nhưng không thể được định vị bởi một thiết bị cảm biến thuần túy. Ví dụ: Sức khỏe tim mạch, Độ bạo lực, Niềm tự hào, và quan trọng nhất trong AI: Khái niệm "Sự Lừa dối" (Deception) hay "Nịnh hót" (Sycophancy).

## 2. Tiết Thiết Lập Cấu Trúc (Methodology)

### 2.1. Cấu Kiến Bức Tranh Tổng Thể Bằng Ráp Nối Phương Trình
Mục đích của Cơ học Giải diễn (Mechanistic Interpretability) không bao giờ là việc đọc hiểu cấu trúc Nơ-ron độc lập (Manifest). Thay vào đó, mục tiêu là sử dụng một hàm Biến đổi (Transformation matrix) lên các vector Biến Hiển Ngôn để trích xuất ra Vector Tiềm Ẩn (Latent Vector).

Phương trình tổng quát cho việc suy diễn này có dạng:

$$

Latent\_Knowledge = Function(Weights, \ Activation\_Patterns\_of\_Neurons)

$$


Trong đó, hàm $Function()$ là sự Kết hợp trọng số tuyến tính (Linear weighted combination) hoặc một biến đổi màng phi tuyến tính, tùy thuộc vào bài toán.

### 2.2. Sự Đổ Vỡ Tuyến Tính (Imperfect Correlations)
Tương tự như Tâm lý học, nơi bài kiểm tra tính cách (Manifest) thường không phải là phản ánh chuẩn tắc 100% của Khí chất Extraversion (Latent) bên trong não bộ, Cơ học Giải diễn vấp phải Nghịch lý Tính Tương quan Kém. Mô hình có thể mang lại kết quả "Ánh mắt (Gaze)" tập trung vào ống kính camera với số điểm 10/10, nhưng "Sự tập trung" (Attention) của sinh thể lại ở mức $\approx 0$. 

Điều này cũng đúng với AI: Model có thể cho ra kết quả Logit Output 99% phù hợp với khái niệm "Đồng ý" (Manifest), nhưng bản thể Latent bên trong nó đang chạy một cụm Vi não được thiết kế để "Lừa Dối" (Deception mode). Đây là sự đe dọa sinh tử cho AI Safety.

---

## 3. Khảo Sát Phương Lý (Analysis)

Việc khai thác Mạch Tiềm Ẩn (Latent Circuit) dựa rập khuôn vào phương thức gom cụm Tế bào hiển môn (Manifest neurons) đã vấp phải giới hạn (như chứng minh từ sự thất bại của Thuật toán T-SNE đối với sự phân mảnh Circuit ngữ pháp). Do đó, giới nghiên cứu AI đã chuyển dịch ứng dụng sang các thiết chế Hàm Tối Ưu không gian Latent đa chiều cực kỳ hiện đại:
- **Phân tích chiều gốc (PCA) / Phân rã giá trị ảo (SVD):** Cơ bản cho các mô hình nhỏ.
- **Autoencoders (Đặc biệt là Sparse Autoencoders - SAE):** Tự xé nhỏ và nén Vector biểu diễn để lọc lấy các tính năng phi cấu trúc trong siêu không gian đa chiều.
- **Phân rã Eigen suy rộng (Generalized Eigen-Decomposition - GED):** Dò tìm các Điểm cộng hưởng quang phổ thay vì Tế bào cơ học vật lý.

---

## 4. Kết Luận
Việc nỗ lực trích xuất các Biến số Tiềm Ẩn từ các Số liệu Hiển ngôn là bài toán khó bậc nhất, luôn tồn tại rủi ro về sai lệch suy diễn không thể đo đạc do "thực thể Tiềm ẩn đó nằm ngoài vùng tiếp cận vật lý". Đặc biệt trong AI Safety, khả năng diễn giải Latent là vũ khí độc quyền để truy thu các khái niệm nguy hiểm mà mô hình LLM đã tự động tích lũy (Lừa lọc, Cảo giác, Rối loạn phân ly). Trong các báo cáo kế tiếp, cơ chế trích xuất Sparse Autoencoder và Generalized Eigendecomposition sẽ được làm rõ về mặt hình thái số học.

---

## Tài Liệu Tham Khảo (Citations)
1. Thuyết Biến Trừu Tượng tại `aero_LLM_07_Latent vs. manifest variables.md`. Giải trình sự chuyển đổi vị trí từ dữ liệu Manifest (Như Activations Neurons, Next-token Logits) sang hàm học thuyết Latent (Deception, Concept Abstraction).
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
| 📌 **[Biến Tiềm Ẩn (Latent) Và Biến Hiển Ngôn (Manifest) Trong Giải Diễn AI](aero_llm_07_latent_vs_manifest_variables.md)** | [Xem bài viết →](aero_llm_07_latent_vs_manifest_variables.md) |
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
