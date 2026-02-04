# Lecture 5: LLM Tuning (SFT & Parameter Efficient) 🎛️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này tập trung vào giai đoạn sau Pre-training: Supervised Fine-Tuning (SFT) để biến mô hình thành trợ lý, và các kỹ thuật Fine-tuning hiệu quả (PEFT/LoRA).

---

## 📚 Mục Lục
1. [Supervised Fine-Tuning (SFT)](#1-supervised-fine-tuning-sft)
2. [Instruction Tuning](#2-instruction-tuning)
3. [Dữ liệu cho SFT](#3-dữ-liệu-cho-sft)
4. [Parameter-Efficient Fine-Tuning (PEFT)](#4-parameter-efficient-fine-tuning-peft)
5. [LoRA (Low-Rank Adaptation)](#5-lora-low-rank-adaptation)
6. [QLoRA (Quantized LoRA)](#6-qlora-quantized-lora)

---

## 1. Supervised Fine-Tuning (SFT)
Sau Pre-training, mô hình giống như một "con vẹt thông thái" - biết rất nhiều nhưng chỉ biết dự đoán từ tiếp theo chứ không biết trả lời câu hỏi hay làm theo lệnh.

**Mục tiêu của SFT:**
*   Dạy mô hình cách **hành xử** (Behavior) mong muốn.
*   Biến Base model -> Chat/Instruct model.

**Quy trình:**
*   Input: Các cặp câu hỏi - câu trả lời (Prompt - Response) chất lượng cao.
*   Loss: Vẫn dùng Next token prediction, nhưng chỉ tính loss trên phần câu trả lời (Response), không tính trên phần câu hỏi (Prompt).

---

## 2. Instruction Tuning
Là một dạng của SFT, tập trung vào việc dạy mô hình tuân theo các chỉ dẫn (instructions) đa dạng.

*   **Tác vụ:** Tóm tắt, dịch, viết code, giải toán, viết thơ, danh sách...
*   **Zero-shot Generalization:** Sau khi được Instruction Tuning trên nhiều tác vụ, mô hình có khả năng thực hiện cả những tác vụ mới mà nó chưa từng thấy trong quá trình training (nhờ khả năng suy luận tổng quát).

---

## 3. Dữ liệu cho SFT
Chất lượng quan trọng hơn số lượng ("Quality is King").

*   **Nguồn dữ liệu:**
    *   *Human-generated:* Do con người viết (đắt đỏ, chất lượng cao).
    *   *LLM-generated (Synthetic data):* Dùng mô hình mạnh (GPT-4) để tạo dữ liệu training cho mô hình nhỏ hơn. (Rẻ, nhanh, nhưng cần kiểm soát chất lượng).
*   **Quy mô:** Chỉ cần hàng nghìn đến hàng trăm nghìn mẫu (ít hơn nhiều so với hàng tỷ tokens của Pre-training).
*   **Ví dụ:** Dataset phổ biến: Alpaca, Vicuna, Lima.

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)
Fine-tuning toàn bộ mô hình (Full Fine-tuning) rất tốn kém (cần VRAM gấp nhiều lần kích thước mô hình để lưu Optimizer states).

**PEFT:** Chỉ cập nhật một phần nhỏ tham số hoặc thêm các module nhỏ vào mô hình, giữ đông cứng (freeze) phần lớn trọng số gốc.

**Lợi ích:**
*   Giảm yêu cầu VRAM.
*   Tránh hiện tượng "Catastrophic Forgetting" (Quên kiến thức cũ).
*   Dễ dàng chia sẻ các Adapter nhỏ (vài chục MB) thay vì cả mô hình GB.

---

## 5. LoRA (Low-Rank Adaptation)
Kỹ thuật PEFT phổ biến nhất hiện nay.

**Ý tưởng:**
Thay vì cập nhật trực tiếp ma trận trọng số $W$ (kích thước $d \times d$), ta cập nhật thông qua 2 ma trận nhỏ $A$ và $B$:
$$ W' = W + \Delta W = W + BA $$
Trong đó:
*   $B$: kích thước $d \times r$
*   $A$: kích thước $r \times d$
*   $r$ (rank): rất nhỏ (ví dụ 8, 16, 64) so với $d$.

**Kết quả:** Số lượng tham số cần train giảm hàng nghìn lần, nhưng hiệu quả tương đương Full Fine-tuning.

---

## 6. QLoRA (Quantized LoRA)
Kết hợp Quantization và LoRA để train mô hình lớn trên GPU nhỏ.

*   **4-bit NormalFloat (NF4):** Một kiểu dữ liệu mới tối ưu cho trọng số phân phối chuẩn của Neural Network.
*   **Double Quantization:** Quantize cả các hằng số quantization để tiết kiệm thêm bộ nhớ.
*   **Paged Optimizers:** Sử dụng CPU RAM để offload optimizer states khi GPU bị tràn bộ nhớ (OOM).

-> Cho phép train mô hình 65B parameters trên một GPU 48GB.

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
