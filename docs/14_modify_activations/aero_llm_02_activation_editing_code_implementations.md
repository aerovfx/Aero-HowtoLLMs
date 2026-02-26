
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [14 modify activations](index.md)

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
# Các Chế Độ Sửa Đổi Hoạt Hóa Cơ Học (Activation Editing Implementations)

## Tóm tắt (Abstract)
Báo cáo này tập trung vào kĩ thuật mã nguồn thực hành Thiết Lập Ma Trận Điều Hướng (Activation Editing) thông qua công cụ Forward Hook trên PyTorch. Các phương án tiếp cận dao động từ can thiệp ghi đè cứng (Hard-coded overwrite) đến một hệ thống Dictionary mềm dẻo cho phép ngắt mở luồng Tín hiệu tuỳ ý theo Layer trong quá trình Suy luận (Inference). Đặc biệt, thông qua chuỗi 6 thí nghiệm, bài viết lý giải về giới hạn Cắt Vùng Tensor (Tensor View) của PyTorch, từ đó định hình cú pháp Đóng gói Nhân Bản (`clone()`) và Khâu Nối (`cat()`) nhằm tạo ra Màng Lọc Tín Hiệu Nhân Quả không sụp đổ bộ nhớ.

---

## 1. Mở Đầu (Introduction)
Như đã đề cập ở Phần Dẫn nhập, nếu Cấu trúc Đọc Quan sát (Observational) chỉ cần đăng ký Hook để lưu giá trị $\text{tensor}$, thì Can Thiệp Nhân Quả (Causal Manipulation) đòi hỏi Hook phải TRẢ VỀ (Return) một Ma trận Output mới ép đè lên luồng chảy của Thuật Toán. 

Cú pháp tổng quát của Phẫu thuật Forward Hook:
```python
def my_hook(module, inputs, output):

$$
modified_output = output.clone()
$$

    # [Inject, Zero-out, hay Scale tùy ý]
    return modified_output
Khi Pytorch quét thấy Hook ném một Object mới về, nó sẽ ép đè lên Biến $\text{output}$ nguyên mẫu nếu hai Object này có Dimensions tuyệt đối khớp nhau. Sự ép đè này mở ra quyền năng sinh-sát đối với bất kỳ Token hay Mạch Activation nào.

---

## 2. Tiết Thiết Kế Thí Nghiệm Cơ Học (Methodology & Implementations)

### Thí Nghiệm 1 & 3: Cạm Bẫy Của Tensor Slices và Kế Hoạch .Clone()
Thống kê Cơ bản: Khi Trích Xuất Attention Block $QKV$, ta thường tách `output` thành cụm độc lập `[Q, K, V]`.
- **Sai Lầm Ở Thí Nghiệm 1:** Nếu thiết lập: `Q[0, 4, :10] = 0` rồi `return torch.cat([Q, K, V], dim=-1)`. Trình biên dịch sẽ báo lỗi *Vô hiệu view()* (RuntimeError: modified in-place view). PyTorch bảo mật không cho phép Sửa đổi một Vùng Nhìn Phễu (View) đang tham chiếu từ Ma trận Cốt lõi của Mạng.
- **Biện Pháp Khắc Phục (Thí Nghiệm 3):** Phải Tạo Bản Sao Lập Trình (Clone) cho Ma trận. Cú pháp: `Q_mod = Q.clone()`. Thao tác Zero-out trên `Q_mod` là hoàn toàn vô can với Bộ đệm Root, và sau khi Concatenate, hàm sẽ nhả ra chuỗi Ghi-đè an toàn. 

### Thí Nghiệm 2: Trỏ Trực Tiếp Theo Mốc Dimension (Direct Indexing)
Nếu không muốn tách/khâu như trên, ta thao tác thẳng vào Block Gốc dựa trên độ Dài Chiều kích (Embedding Dimension).
Với GPT-2 Small, $d\_model = 768$. Ma trận $QKV$ dài $768 \times 3 = 2304$. 
- Để Tịt ngòi $10$ Vector của lớp **Q** cho Token Index số $4$: `output[0, 4, 0:10] = 0`
- Để Tịt ngòi $10$ Vector của lớp **K** cho Token Index số $4$: `output[0, 4, 768 : 768+10] = 0`
Phương pháp này chọc ngang vùng Core Tensor nên không bị lỗi View, nhưng đòi hỏi phải Hard-code phép Offset chỉ số rất mệt mỏi.

### Thí Nghiệm 4 & 5: Biến Tổng Cục & Bộ Điều Khiển Cóc (Layer Isolation)
Mục tiêu: Đặt Hook lên 12 Lớp Transformer, Thu hoạch *Toàn bộ* QKV của 12 lớp, nhưng **Chỉ Bóp Méo Duy Nhất Lớp thứ 3**.
- Dùng `if layer_num == 3: ...` chặn cổng điều tiết.
- Không gán ép giá trị rỗng $0$ (Rất phí khoa học do nhiễu bất thường), mà Khởi tạo Biến Dữ liệu Toàn Cục (Global Array `q_to_replace = torch.linspace(-1, 0.7)`). Ta có thể đổi Biến Đầu Vào (Injection Vector) từ không gian thí nghiệm mà không cần rã Hook ra định nghĩa lại.

### Thí Nghiệm 6: Bảng Điều Phối Dictionary (The Dictionary Injection Board)
Đây là Cảnh giới Linh hoạt nhất.
- Tạo một `replacement_dict = {3: zeros_tensor, 11: specific_vector}`
- Logic Hàm sẽ là: `if layer_num in dict.keys(): output = dict[layer_num]`
Với mô hình này, ta có thể bật / tắt nhiễu từng phần, đảo Cột tín hiệu giữa Block $3 \to Block\ 11$ cực kỳ cơ động chỉ với 1 thao tác Gán (Assign), tạo lợi thế tốc độ trong các Test Suite Khổng Lồ.

---

## 3. Lời Khuyên (Best Practices)
Kinh nghiệm sâu sắc với Causal Manipulation là "Mắc cài In. Ra. In". 
Quá trình Forward Pass chạy với tốc độ hàng phần nghìn giây và không cho tương tác Real-time giữa chừng. Nên khi mới Code Hook:
- Hãy Bơm Tịt $0$ (Zero-out check) ở mọi lúc để đo đếm Đầu vào/Đầu ra đã Khớp kích thước hay chưa.
- Lợi dụng Hàm `print()` đẩy Shape vào màn hình Console để thấy Dòng chảy Dữ liệu đang cắt ở đâu.

## Tài liệu Tham khảo (Citations)
1. 06 Quy trình Thao Túng Pytorch Causal Framework - Lược sử từ `aero_LLM_02_Activation editing Code implementations.md`. So sánh đối chứng Lỗi In-place View Tensor và giải pháp Khối Dictionary.
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [Dẫn Nhập Về Diễn Giải Cơ Học Nhân Quả (Causal Mechanistic Interpretability)](aero_llm_01_introduction_to_causal_mech_interp.md) | [Xem bài viết →](aero_llm_01_introduction_to_causal_mech_interp.md) |
| 📌 **[Các Chế Độ Sửa Đổi Hoạt Hóa Cơ Học (Activation Editing Implementations)](aero_llm_02_activation_editing_code_implementations.md)** | [Xem bài viết →](aero_llm_02_activation_editing_code_implementations.md) |
| [Thử thách Lập trình: Thay thế Hoạt hóa Attention, MLP và Hidden States](aero_llm_03_codechallenge_replacing_attention_mlp_and_hidden_states.md) | [Xem bài viết →](aero_llm_03_codechallenge_replacing_attention_mlp_and_hidden_states.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
