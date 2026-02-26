
<!-- Aero-Navigation-Start -->
[🏠 Home](../../index.md) > [17 modifying mlp](../index.md)

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](../../index.md)
- [📚 Module 01: LLM Course](../../01_llm_course/index.md)
- [🔢 Module 02: Tokenization](../../02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](../../04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](../../07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](../../19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](../../20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# Thay thế Trung vị Nối tiếp các Neurons trong Lớp MLP (Successive Median-Replacement of MLP Neurons)

## Tóm tắt (Abstract)
Báo cáo này nghiên cứu tác động của việc can thiệp vào tầng mở rộng (Expansion Layer) của khối MLP trong mô hình GPT-2 Large. Do số lượng neuron cực lớn (hơn 5000), việc kiểm tra từng neuron là không khả thi. Nghiên cứu đề xuất một phương pháp dựa trên thống kê mô tả: thay thế một tỷ lệ phần trăm các neuron hoạt động mạnh nhất bằng giá trị trung vị (median) của toàn bộ quần thể. Một phát hiện đáng kinh ngạc là mô hình thể hiện tính bền vững cao đối với các can thiệp diện rộng (từ 10% đến 90%), và chỉ bắt đầu cho thấy sự biến thiên đáng kể khi tỷ lệ can thiệp giảm xuống mức tinh vi (dưới 10%). Điều này củng cố giả thuyết rằng thông tin quan trọng được mã hóa bởi một nhóm rất nhỏ các neurons chuyên biệt.

---

## 1. Mở Đầu (Introduction)
Khối MLP (Multi-Layer Perceptron) đóng vai trò xử lý phi tuyến tính và lưu trữ tri thức trong kiến trúc Transformer. Thách thức lớn nhất khi nghiên cứu MLP là "Sự bùng nổ chiều" (Dimensionality Explosion). Trong GPT-2 Large, mỗi block chứa 5120 neurons MLP. Báo cáo này giới thiệu kỹ thuật can thiệp dựa trên ngưỡng hoạt hóa để giải quyết vấn đề quy mô này một cách hiệu quả.

---

## 2. Phương Pháp Thực Nghiệm (Methodology)

### 2.1. Kỹ thuật Median-Replacement
Thay vì triệt tiêu neuron (Zeroing), chúng ta sử dụng giá trị trung vị của tầng đó để thay thế. 
- **Lý do:** Giá trị trung vị đại diện cho mức hoạt động "nền" của tầng, giúp quan sát tác động của việc mất đi các tín hiệu cực đỉnh (peaks) mà không làm xáo trộn hoàn toàn phân phối năng lượng của hệ thống.
- **Quy trình:** 
    1. Trích xuất hoạt hóa MLP từ tầng `c_fc` cho token cuối cùng.
    2. Xác định $p\%$ neurons có giá trị hoạt hóa cao nhất.
    3. Ghi đè các giá trị này bằng trung vị của 5120 neurons trong block đó.

### 2.2. Thử nghiệm "Tỷ lệ Ripple" (Ripple-rate Experiment)
Thực hiện vòng lặp thay thế với các tỷ lệ nối tiếp: $10\%, 20\%, \dots, 90\%$. Biến phụ thuộc là sai lệch logit của token mục tiêu (ví dụ từ "night" trong câu "It was a dark and stormy...").

---

## 3. Kết Quả Và Phân Tích (Results & Analysis)

### 3.1. Tính không nhạy cảm đối với thang đo lớn
- **Phát hiện:** Biểu đồ kết qua cho thấy các đường biểu diễn của tỷ lệ 10% và 90% gần như đè khít lên nhau tại hầu hết các tầng.
- **Ý nghĩa:** Điều này cực kỳ phản trực giác. Nó gợi ý rằng một khi bạn đã vô hiệu hóa "nhóm lõi" (core group) của các neurons mang thông tin, việc vô hiệu hóa thêm hàng ngàn neurons khác cũng không làm thay đổi thêm dự đoán của mô hình.

### 3.2. Hiệu ứng Ngưỡng (Threshold Effect)
Khi giảm tỷ lệ can thiệp xuống mức siêu nhỏ ($0.2\% - 4.5\%$):
- Sự biến thiên bắt đầu xuất hiện rõ rệt.
- **Kết luận:** Hầu hết thông tin điều chỉnh vector embeddings trong residual stream chỉ được mang bởi khoảng $1\% - 2\%$ neurons hoạt động mạnh nhất. Phần lớn các neurons còn lại đóng vai trò dự phòng hoặc chỉ mang các đóng góp cực nhỏ (infinitesimal contributions) mà không làm thay đổi logit đầu ra một cách đáng kể.

### 3.3. Tác động của Tầng đầu tiên (Layer 0)
Can thiệp tại Transformer Block 0 cho thấy tác động "catastrophic" (thảm khốc). Mô hình hoàn toàn mất khả năng dự đoán từ ngữ đơn giản. Điều này chứng tỏ các tầng MLP đầu tiên đóng vai trò "cổng thông tin" sống còn để chuyển đổi embeddings thô thành các biểu diễn có ngữ cảnh.

---

## 4. Thảo Luận: Các Neurons chuyên biệt
Sự khác biệt giữa việc can thiệp 10% và 90% cho thấy tri thức trong LLM được phân bổ theo quy luật "ít nhưng tinh" (sparse coding). Một số ít neurons mang thông tin ngữ nghĩa mạnh mẽ, trong khi phần còn lại tạo thành một "nền văn hóa" ổn định. Việc nghiên cứu sâu hơn cần tập trung vào việc cô lập $1\%$ neurons quyền lực này thông qua các phương pháp thống kê suy diễn.

---

## 5. Kết Luận
Phương pháp thay thế trung vị nối tiếp đã chứng minh rằng khối MLP không phải là một khối đồng nhất. Mặc dù có kích thước khổng lồ, chức năng thực tế của nó tập trung vào một nhóm tiểu số các neuron. Những phát hiện này mở đường cho việc xây dựng các bản đồ mạch thần kinh (circuits) tinh gọn hơn trong tương lai.

---

## Tài liệu tham khảo (Citations)
1. Thí nghiệm Median-Replacement trên GPT-2 Large dựa trên `aero_LLM_01_Successive median-replacement of MLP neurons.md`. Phân tích tính bền vững của MLP và mật độ thông tin nén.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| 📌 **[Thay thế Trung vị Nối tiếp các Neurons trong Lớp MLP (Successive Median-Replacement of MLP Neurons)](aero_llm_01_successive_median_replacement_of_mlp_neurons.md)** | [Xem bài viết →](aero_llm_01_successive_median_replacement_of_mlp_neurons.md) |
| [Cắt bỏ Tiệm cận các Neurons MLP trên cơ sở Thống kê (Statistics-based Lesioning of MLP Neurons)](aero_llm_02_statistics_based_lesioning_mlp_neurons.md) | [Xem bài viết →](aero_llm_02_statistics_based_lesioning_mlp_neurons.md) |
| [Thử thách Lập trình: Hồ sơ Phân tầng của các T-lesions trong MLP (Laminar Profile of MLP T-lesions)](aero_llm_03_codechallenge_laminar_profile_of_mlp_t_lesions.md) | [Xem bài viết →](aero_llm_03_codechallenge_laminar_profile_of_mlp_t_lesions.md) |
| [Khám phá việc Loại bỏ Không gian con trong MLP (Explorations in Subspace Removal)](aero_llm_04_explorations_in_subspace_removal.md) | [Xem bài viết →](aero_llm_04_explorations_in_subspace_removal.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
