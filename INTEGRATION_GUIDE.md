# 🛠️ Hướng dẫn tích hợp LLM Visualization Helpers

Tài liệu này hướng dẫn cách sử dụng bộ công cụ hỗ trợ trực quan hóa LLM, giải quyết lỗi hiển thị tiếng Việt và cung cấp thư viện công thức toán học.

## 1. Xử lý tiếng Việt trong môi trường 3D (WebGL/WebGPU)

### Vấn đề:
Font atlas mặc định chỉ hỗ trợ ASCII, khiến các ký tự có dấu như "ệ", "ố", "ữ" hiển thị thành "□" hoặc bị ẩn.

### Giải pháp ngắn hạn (Quick Fix):
Sử dụng `removeVietnameseTones` để chuyển văn bản sang không dấu trước khi gửi vào buffer rendering.

```typescript
import { removeVietnameseTones } from './src/utils/vietnamese';

// Trong hàm render labels 3D
const labelText = removeVietnameseTones("Trọng số đầu vào"); 
// Kết quả: "Trong so dau vao" (Hiển thị an toàn trong 3D)
```

### Giải pháp dài hạn (Proper Fix):
Đã cập nhật `create-font-atlas.jsm` để bao gồm bảng mã tiếng Việt. Chạy lệnh sau để tạo lại atlas:
```bash
node create-font-atlas.jsm
```

---

## 2. Sử dụng thư viện công thức toán học

Chúng tôi cung cấp hơn 40 công thức toán học dưới dạng LaTeX (cho UI) và ASCII Math (cho nhãn 3D).

### Ví dụ tích hợp:

```typescript
import { getFormula } from './src/utils/latex-formula-generator';

const formula = getFormula('ATTENTION_BASIC');

// 1. Hiển thị trong 3D (dùng ASCII Math)
render3DLabel(formula.asciiMath, position); 

// 2. Hiển thị trong UI (dùng LaTeX với KaTeX)
<div className="latex-container">
  {formula.latex}
</div>
```

## 3. Danh sách các Categories công thức
- `attention`: Các công thức về cơ chế chú ý.
- `position`: Mã hóa vị trí (Positional Encoding).
- `ffn`: Feed-forward networks và hàm kích hoạt.
- `normalization`: LayerNorm, RMSNorm.
- `loss`: Cross Entropy, Perplexity.
- `sampling`: Softmax, Temperature, Top-K/P.
- `rlhf`: Reward model, PPO loss.

---

---

## 5. Giải pháp hiển thị toán học tối ưu: Hybrid Math Rendering

Đây là giải pháp khuyến nghị nhất cho project, kết hợp 3 lớp hiển thị để đạt cân bằng hoàn hảo giữa hiệu năng và thẩm mỹ.

### Cách sử dụng:
Yêu cầu cài đặt: `npm install katex react-katex html2canvas`

```tsx
import { HybridMathRenderer } from './src/llm/components/HybridMathRenderer';

// Tích hợp vào Architecture Design stage
<HybridMathRenderer 
  formulaKey="ATTENTION_WEIGHTS" 
  mode="all" 
/>
```

### Ưu điểm:
- **Scene 3D**: hiển thị ASCII nhanh và nhẹ.
- **Tooltip 2D**: hiển thị KaTeX sắc nét, đầy đủ định dạng.
- **Billboard**: tự động tạo texture cho các công thức quan trọng.

Xem chi tiết tại: [HYBRID_MATH_GUIDE.md](./HYBRID_MATH_GUIDE.md)

---

## 6. Checklist triển khai
- [x] Đã cập nhật `font-atlas.png` hỗ trợ tiếng Việt.
- [x] Đã tích hợp `removeVietnameseTones` vào `fontRender.ts`.
- [x] Đã khởi tạo thư viện công thức toán học (hơn 60 công thức).
- [x] Đã cài đặt KaTeX và html2canvas.
- [x] Đã triển khai `HybridMathRenderer.tsx`.
- [ ] Kiểm tra hiển thị trên thiết bị di động.

---
**release v1.0.1** - Cập nhật Hybrid Rendering & Fixed Formulas.
