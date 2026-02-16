
# 🚀 Aero-HowtoLLMs: Lộ Trình Toàn Diện Master LLM & Visualization

> **Dự án học tập chuyên sâu từ A-Z về Large Language Models (LLM), tích hợp Interactive 3D Visualization và Hệ thống Tài liệu tiếng Việt.**

[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Tech: Next.js 13](https://img.shields.io/badge/Tech-Next.js%2013-blue.svg)]()
[![Content: 100% Vietnamese](https://img.shields.io/badge/Content-Vietnamese%20Localized-red.svg)]()

---

## 🌟 ĐIỂM NHẤN DỰ ÁN

### 1. Interactive 3D LLM Visualizer (GPT-4 ↔ MoE) 👁️
Mô tản trực quan sống động kiến trúc Transformer với các tính năng:
- **100% Tiếng Việt:** Toàn bộ Walkthrough và Commentary đã được Việt hóa.
- **Kiến trúc MoE (Mixture of Experts):** Trực quan hóa Router và Grid Expert (2x4).
- **Deep Dive Components:** Tương tác với Token Embeddings, Multi-Head Attention, MLP, Residual Connections, và Softmax.
- **Hiệu ứng Animation:** Luồng dữ liệu, kích hoạt Expert top-K, và quá trình sinh token.

👉 **Chạy Visualizer:** `npm run dev` (truy cập `localhost:3002`)

### 2. Hệ Thống Tài Liệu Chuyên Sâu (Docs Suite) 📚
Hơn 100 file Markdown được biên soạn khoa học, bao gồm các chủ đề:

| Module | Nội Dung |
|--------|----------|
| **[Stanford Course](docs/LLM_Course/README.md)** | Chuyển ngữ và bổ sung từ Stanford CME 295 (5 Chương cốt lõi). |
| **[Pre-training & Arch](docs/pretraining/)** | Xây dựng GPT từ con số 0, xử lý dữ liệu và Scaling Laws. |
| **[Fine-tuning Series](docs/Fine-tune%20pretrained%20models/README.md)** | 23 chương thực chiến: LoRA, PEFT, và series Alice vs Edgar. |
| **[RAG & Applications](docs/rag/)** | Triển khai RAG với FastAPI, Qdrant và Ollama. |
| **[AI Safety & Interpretability](docs/AI%20safety%20and%20mechanistic%20interpretability/)** | Phân tích cơ chế và an toàn AI. |

---

## 🗺️ LỘ TRÌNH HỌC TẬP (ROADMAP)

### 🟢 Giai đoạn 1: Nền Tảng (Fundamentals)
- Tìm hiểu kiến trúc Transformer gốc qua **[Sơ đồ trực quan](docs/COMPLETION_VISUALIZATION_AND_CHAPTERS.md)**.
- Học 5 trụ cột của LLM: Architecture, Data, Loss, Evaluation, Systems.
- **[Xem tài liệu Overview](docs/LLM_Course/LectureStanford/aero_LLM_00_Overview.md)**.

### 🟡 Giai đoạn 2: Huấn Luyện & Cấu Trúc (Pre-training)
- **[BuildGPT](docs/buildGPT/)**: Từng bước xây dựng mô hình trong code.
- Xử lý các vấn đề số học (Numerical stability), Normalization, và Optimization.

### 🟠 Giai đoạn 3: Tinh Chỉnh & Thích Nghi (Fine-tuning)
- Thực hiện các **CodeChallenge** thực tế:
    - Tinh chỉnh phong cách văn học (Alice in Wonderland vs Edgar Allan Poe).
    - Định lượng hiệu quả bằng mô hình phân loại (BERT integration).
    - Các kỹ thuật tối ưu: Freezing Attention, PEFT, LoRA.
- **[Xem danh mục Fine-tuning](docs/Fine-tune%20pretrained%20models/README.md)**.

### 🔴 Giai đoạn 4: Ứng Dụng Nâng Cao (Advanced)
- Triển khai **RAG (Retrieval Augmented Generation)** cho dữ liệu nội bộ.
- Xây dựng AI Agents và thực hiện Instruction Tuning.
- **[Xem lộ trình Hybrid AI](docs/roadmapHybridAI.md)**.

---

## 🛠️ CÀI ĐẶT & SỬ DỤNG

### Yêu cầu hệ thống
- Node.js 18+
- RAM: Tối thiểu 8GB (để chạy Visualizer mượt mà)

### Các bước cài đặt
1. Clone repository:
   ```bash
   git clone https://github.com/aerovfx/Aero-HowtoLLMs.git
   ```
2. Cài đặt dependency:
   ```bash
   npm install
   ```
3. Khởi chạy môi trường phát triển (Visualizer):
   ```bash
   npm run dev
   ```
   *Mở trình duyệt tại: http://localhost:3002*

---

## 📊 THỐNG KÊ DỰ ÁN
- **Số lượng tài liệu:** ~1,800 dòng nội dung chuyên sâu được cập nhật gần nhất.
- **Ngôn ngữ:** 100% hỗ trợ tiếng Việt (Localized).
- **Tính năng Visual:** GPT-4 ↔ MoE (Mixture of Experts).

---

## 🤝 ĐÓNG GÓP & LIÊN HỆ
Dự án được biên soạn và duy trì bởi **Pixibox** phục vụ cộng đồng AI Việt Nam.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

---
*Cập nhật lần cuối: 16/02/2026*
