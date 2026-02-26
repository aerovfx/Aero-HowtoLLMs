
<!-- Aero-Navigation-Start -->
[🏠 Home](../index.md) > [07 fine tune pretrained models](index.md)

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
# Lưu Trữ và Tải Lại Mô Hình Học Sâu Trong PyTorch và Hugging Face: Phương Pháp, Cấu Trúc và Đánh Giá

## Tóm tắt

Bài viết này trình bày các phương pháp lưu trữ và tải lại mô hình học sâu trong môi trường PyTorch và hệ sinh thái Hugging Face. Dựa trên tài liệu thực nghiệm , nghiên cứu phân tích cấu trúc dữ liệu mô hình, cơ chế tuần tự hóa (serialization), quy trình khôi phục tham số, và kỹ thuật đóng gói mô hình. Các công thức toán học được sử dụng nhằm mô hình hóa quá trình cập nhật và bảo toàn tham số. Kết quả cho thấy việc lưu – tải mô hình đúng cách đóng vai trò then chốt trong tái sử dụng, triển khai và nghiên cứu AI.

---

## 1. Giới thiệu

Trong quá trình huấn luyện mô hình học sâu, việc không lưu trữ kết quả sẽ dẫn đến mất toàn bộ tham số khi phiên làm việc kết thúc. Điều này đặc biệt quan trọng trong môi trường điện toán đám mây như **Google Colab**.

Theo tài liệu , tác giả trình bày cách lưu và tải lại mô hình ngôn ngữ GPT-2 bằng công cụ của **Hugging Face** và **PyTorch**.

Mô hình minh họa chính trong nghiên cứu là **GPT-2**, một mô hình ngôn ngữ tiền huấn luyện phổ biến.

Mục tiêu nghiên cứu:

* Phân tích cấu trúc dữ liệu mô hình,
* Mô tả cơ chế lưu – tải tham số,
* So sánh phương pháp Hugging Face và PyTorch,
* Đánh giá hiệu quả bảo toàn mô hình.

---

## 2. Cơ sở lý thuyết

### 2.1. Biểu diễn tham số mô hình

Một mô hình học sâu được đặc trưng bởi tập tham số:

\theta = {W_1, W_2, \dots, W_L, b_1, b_2, \dots, b_L}

Trong đó:

* $W_l$: ma trận trọng số,
* $b_l$: vector bias,
* $L$: số lớp.

Toàn bộ tập $\theta$ cần được lưu trữ để tái tạo mô hình.

---

### 2.2. Quá trình huấn luyện

Tham số được cập nhật theo gradient descent:

\theta_{t+1} = \theta_t-\eta\nabla_\theta\mathcal{L}_t

với:

* $\eta$: learning rate,

$$
* \mathcal{L}: hàm mất mát.
$$

Mục tiêu của việc lưu mô hình là bảo toàn $\theta_T$ tại thời điểm hội tụ.

---

### 2.3. State Dictionary

Trong PyTorch, trạng thái mô hình được biểu diễn bởi:

\text{state_dict}={\theta_i}_{i=1}^{P}

với $$P( là số tensor tham số.

---

## 3. Cấu trúc lưu trữ mô hình Hugging Face

### 3.1. Định dạng thư mục

Theo , mô hình Hugging Face không được lưu dưới dạng một file duy nhất mà là một thư mục gồm:

* `config.json`,
* `tokenizer.json`,
* `model.safetensors`,
* `version.txt`.

Cấu trúc:

)
\mathcal{F}={f_1,f_2,\dots,f_k}

Trong đó f_k chứa toàn bộ tham số. --- ### 3.2. File trọng số File `model.safetensors` chứa ma trận:

W\in\mathbb{R}^{d\times d'}

Dung lượng xấp xỉ:

S\approx 4\times P \text{ bytes}

với P( là số tham số dạng float32. Ví dụ GPT-2 small: )

$$
S\approx 474\text{ MB} . --- ### 3.3. Lệnh lưu mô hình Phương thức:
$$

\text{model.save_pretrained(path)}

$$
Thực hiện ánh xạ: \theta \rightarrow \mathcal{F}_{path} --- ## 4. Chỉnh sửa và kiểm chứng mô hình ### 4.1. Thao tác thay đổi embedding Theo tài liệu, embedding được thay bằng vector 1: E_{ij}=1,\ \forall i,j Thay vì: E_{ij}\sim \mathcal{N}(0,\sigma^2) Điều này giúp kiểm tra tính đúng đắn khi tải lại mô hình. --- ### 4.2. So sánh tham số Trước và sau khi chỉnh sửa: \Delta E = E_{new}-E_{old} Nếu:
$$

|\Delta E|_F>0

$$
⇒ mô hình đã thay đổi. --- ### 4.3. Khôi phục mô hình Sử dụng:
$$

\text{from_pretrained(path)}

$$
Tái tạo: \theta_{load}\approx\theta_{save} --- ## 5. Lưu trữ bằng PyTorch ### 5.1. Lưu state dictionary Với PyTorch:
$$

\text{torch.save(state_dict, file.pt)}

$$
Biểu diễn:
$$

\theta \rightarrow file.pt

$$
Khác với Hugging Face, phương pháp này chỉ tạo một file. --- ### 5.2. Tải lại mô hình \theta \leftarrow \text{torch.load(file.pt)} và:
$$

\text{model.load_state_dict}(\theta)

$$
Giúp khôi phục tham số. --- ### 5.3. Tính toàn vẹn tham số Sai số khôi phục: \varepsilon=|\theta_{load}-\theta_{orig}|_2 Lý tưởng: \varepsilon\approx 0 --- ## 6. Đóng gói và di chuyển mô hình ### 6.1. Nén thư mục Theo , sử dụng: \text{zip}(\mathcal{F})\rightarrow file.zip Tỷ lệ nén: r=\frac{S_{zip}}{S_{raw}} Thông thường: r\approx 0.8-0.9 với mô hình lớn. --- ### 6.2. Giải nén file.zip \rightarrow \mathcal{F}' Sao cho:
$$

\mathcal{F}'\equiv\mathcal{F}

$$
--- ### 6.3. Di chuyển môi trường Quy trình: 1. Nén mô hình, 2. Tải về máy cá nhân, 3. Upload lên phiên mới, 4. Giải nén, 5. Load mô hình. Đảm bảo: P(\text{lỗi})\approx 0 --- ## 7. Phương pháp đánh giá ### 7.1. So sánh đầu ra Cho input x: y_{old}=f(x;\theta_{old})
$$

y_{new}=f(x;\theta_{load})

Sai lệch:

\delta=|y_{old}-y_{new}|

$$
Nếu \delta\approx0 ⇒ khôi phục thành công. --- ### 7.2. Kiểm tra embedding Trường hợp kiểm chứng bằng vector 1: E_{ij}=1 \Rightarrow \text{mean}(E)=1 Nếu đúng ⇒ tải đúng mô hình. --- ### 7.3. Đánh giá độ ổn định Tính phương sai đầu ra: \sigma^2=\frac{1}{N}\sum(y_i-\bar{y})^2 Mô hình ổn định ⇒ \sigma^2 thấp. --- ## 8. Thảo luận ### 8.1. So sánh hai phương pháp | Tiêu chí      | Hugging Face | PyTorch    | | ------------- | ------------ | ---------- | | Định dạng     | Thư mục      | File       | | Dễ triển khai | Cao          | Trung bình | | Linh hoạt     | Trung bình   | Cao        | | Tính phổ quát | Thấp         | Cao        | --- ### 8.2. Ưu điểm * Bảo toàn tri thức huấn luyện, * Hỗ trợ tái sử dụng, * Thuận tiện triển khai. --- ### 8.3. Hạn chế * Dung lượng lớn, * Phụ thuộc phiên bản, * Khó chuẩn hóa liên thư viện. --- ## 9. Ứng dụng thực tiễn Phương pháp lưu – tải mô hình được ứng dụng trong: * Triển khai hệ thống NLP, * Chia sẻ mô hình nghiên cứu, * Fine-tuning nhiều giai đoạn, * Học tập và giảng dạy AI. Đặc biệt quan trọng trong môi trường cloud:
$$

T_{session}\lt T_{train}