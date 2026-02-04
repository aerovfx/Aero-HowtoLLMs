# Lecture 4: LLM Training - Pre-training 🏋️

> **Tóm tắt từ khóa học Stanford CME 295: Transformers & Large Language Models.**
> Bài giảng này tập trung vào quy trình huấn luyện LLM, từ Pre-training (Tiền huấn luyện) đến các kỹ thuật tối ưu hóa phần cứng (Parallelism, FlashAttention).

---

## 📚 Mục Lục
1. [Quy trình huấn luyện LLM](#1-quy-trình-huấn-luyện-llm)
2. [Pre-training (Tiền huấn luyện)](#2-pre-training-tiền-huấn-luyện)
3. [Luật mở rộng (Scaling Laws)](#3-luật-mở-rộng-scaling-laws)
4. [Tối ưu hóa phần cứng & Bộ nhớ](#4-tối-ưu-hóa-phần-cứng--bộ-nhớ)
5. [FlashAttention ⚡](#5-flashattention-⚡)
6. [Quantization & Mixed Precision Training](#6-quantization--mixed-precision-training)

---

## 1. Quy trình huấn luyện LLM
Huấn luyện LLM không phải là một bước duy nhất mà là một quy trình nhiều giai đoạn (Multi-stage training):

1.  **Pre-training (Tiền huấn luyện):** Học kiến thức tổng quát từ dữ liệu khổng lồ (Internet, sách, code).
    *   *Mục tiêu:* Dự đoán từ tiếp theo (Next token prediction).
    *   *Kết quả:* Base model (Mô hình nền tảng) - biết nhiều nhưng chưa biết làm trợ lý.
2.  **Fine-tuning (Tinh chỉnh):**
    *   *SFT (Supervised Fine-Tuning):* Dạy mô hình làm theo hướng dẫn (Instruction Following).
    *   *RLHF/RLAIF:* Căn chỉnh mô hình theo sở thích con người (Alignment).

---

## 2. Pre-training (Tiền huấn luyện)
Đây là giai đoạn tốn kém nhất (hàng triệu USD).

*   **Dữ liệu (Data):** Common Crawl (Internet), Wikipedia, Reddit, GitHub (Code).
    *   *Quy mô:* Hàng nghìn tỷ tokens (Trillions of tokens).
    *   *Ví dụ:* Llama 3 được train trên 15T tokens.
*   **Thách thức:**
    *   **Cost (Chi phí):** Rất lớn về tiền bạc và năng lượng.
    *   **Knowledge Cutoff (Giới hạn kiến thức):** Mô hình chỉ biết những gì xảy ra trước thời điểm thu thập dữ liệu.
    *   **Hallucination (Ảo giác):** Có thể bịa đặt thông tin.

---

## 3. Luật mở rộng (Scaling Laws)
Làm sao để mô hình thông minh hơn?

*   **Kaplan et al. (2020):** Hiệu năng mô hình tăng theo hàm mũ (power law) khi tăng:
    1.  Kích thước mô hình (Parameters).
    2.  Kích thước dữ liệu (Dataset size).
    3.  Lượng tính toán (Compute).
*   **Chinchilla Scaling Law (Hoffmann et al., 2022):**
    *   Để tối ưu hóa chi phí tính toán, khi tăng gấp đôi kích thước mô hình, cần tăng gấp đôi lượng dữ liệu.
    *   **Công thức vàng:** Số lượng tokens huấn luyện nên gấp khoảng **20 lần** số lượng tham số mô hình.
    *   *Hệ quả:* Nhiều mô hình trước đó (như GPT-3) là "under-trained" (huấn luyện chưa đủ). Các mô hình hiện đại (Llama) có xu hướng nhỏ hơn nhưng train lâu hơn (trên nhiều dữ liệu hơn).

---

## 4. Tối ưu hóa phần cứng & Bộ nhớ
Huấn luyện LLM đòi hỏi bộ nhớ VRAM khổng lồ, vượt xa khả năng của một GPU đơn lẻ (ví dụ H100 80GB).

### Các kỹ thuật song song hóa (Parallelism):
1.  **Data Parallelism (DP):**
    *   Copy mô hình ra nhiều GPU.
    *   Mỗi GPU xử lý một phần dữ liệu (batch) khác nhau.
    *   Đồng bộ hóa gradient sau mỗi bước.
2.  **Model Parallelism (MP):**
    *   **Tensor Parallelism (TP):** Chia nhỏ các ma trận trọng số (weight matrices) để tính toán song song trên nhiều GPU.
    *   **Pipeline Parallelism (PP):** Chia các lớp (layers) của mô hình cho các GPU khác nhau (GPU 1 làm lớp 1-10, GPU 2 làm lớp 11-20...).
3.  **ZeRO (Zero Redundancy Optimizer):**
    *   Tối ưu hóa Data Parallelism bằng cách chia nhỏ Optimizer States, Gradients, và Parameters ra các GPU thay vì mỗi GPU phải giữ một bản copy đầy đủ.

---

## 5. FlashAttention ⚡
**Vấn đề:** Attention tiêu chuẩn có độ phức tạp $O(N^2)$ và tốn rất nhiều thao tác đọc/ghi bộ nhớ (Memory IO) giữa HBM (bộ nhớ chậm, lớn) và SRAM (bộ nhớ nhanh, nhỏ).

**Giải pháp (FlashAttention - Dao et al., 2022):**
*   **Tiling:** Chia ma trận Attention thành các khối nhỏ (tiles) để tính toán hoàn toàn trong SRAM tốc độ cao.
*   **Recomputation:** Chấp nhận tính toán lại một số giá trị trong quá trình backward pass thay vì lưu trữ tất cả (tiết kiệm VRAM, giảm IO).
*   **Kết quả:** Tăng tốc độ huấn luyện 2-4 lần, giảm bộ nhớ sử dụng 10-20 lần, cho phép train với context length dài hơn.

---

## 6. Quantization & Mixed Precision Training
Giảm độ chính xác của số học để tiết kiệm bộ nhớ và tăng tốc độ.

*   **FP32 (Full Precision - 32 bit):** Chuẩn cũ, rất tốn kém.
*   **FP16 / BF16 (Half Precision - 16 bit):** Tiêu chuẩn hiện nay cho training.
    *   **BF16 (Brain Float 16):** Giữ nguyên dải giá trị (range) của FP32 nhưng giảm độ chính xác phần thập phân (mantissa). Ổn định hơn FP16 cho training LLM.
*   **Mixed Precision Training:**
    *   Lưu trữ trọng số chính (Master weights) ở FP32.
    *   Tính toán Forward/Backward ở FP16/BF16.
    *   Cập nhật trọng số ở FP32.

---
*Biên soạn bởi Pixiboss - Dựa trên Stanford CME 295.*
