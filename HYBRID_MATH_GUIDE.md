# 🎯 HƯỚNG DẪN: GIẢI PHÁP HYBRID MATH RENDERING

Giải pháp **Hybrid Math Rendering** được thiết kế để giải quyết bài toán khó nhất trong trực quan hóa AI: Làm thế nào để hiển thị các công thức toán học phức tạp trong không gian 3D mà không làm giảm hiệu năng hệ thống?

---

## 🏗️ KIẾN TRÚC 3 LỚP (3-LAYER ARCHITECTURE)

### 1. Lớp Scene 3D (Simplified ASCII)
- **Mục tiêu**: Tốc độ render tối đa, không gây giật lag (Zero overhead).
- **Cách dùng**: Sử dụng thuộc tính `asciiMath` từ thư viện công thức.
- **Trường hợp**: Cho các nhãn nhỏ, các bước trung gian hoặc khi số lượng công thức trên màn hình vượt quá 100.
- **Ví dụ**: `LN(x) = γ(x-μ)/σ + β`

### 2. Lớp Billboard (High-Quality Textures)
- **Mục tiêu**: Hiển thị các công thức quan trọng nhất với chất lượng sách giáo khoa.
- **Cách dùng**: `html2canvas` render KaTeX thành một texture ảnh và dán lên một mặt phẳng (billboard) trong 3D.
- **Trường hợp**: Các công thức "trái tim" như Attention Score, Softmax Weights.
- **Lưu ý**: Chỉ dùng tối đa 5-10 billboard cùng lúc để tiết kiệm bộ nhớ GPU.

### 3. Lớp Tooltip (Full LaTeX Typography)
- **Mục tiêu**: Cung cấp chi tiết đầy đủ khi người dùng tương tác.
- **Cách dùng**: React + KaTeX overlay trên layer 2D.
- **Trường hợp**: Kích hoạt khi hover hoặc click vào bất kỳ công thức nào.
- **Điểm mạnh**: Độ phân giải vector, copy được text, giải thích chi tiết.

---

## 🚀 CÁCH TÍCH HỢP VÀO PROJECT

### Bước 1: Khai báo Formula Key
Đảm bảo công thức đã có trong `latex-formula-generator.ts`.

### Bước 2: Sử dụng Component
```tsx
import { HybridMathRenderer } from './components/HybridMathRenderer';

// Ví dụ: Hiển thị Attention với đầy đủ 3 lớp
<HybridMathRenderer 
  formulaKey="ATTENTION_BASIC" 
  mode="all" 
  interactive={true} 
/>

// Ví dụ: Chỉ hiển thị text đơn giản cho các bước phụ
<HybridMathRenderer 
  formulaKey="SOFTMAX" 
  mode="simple" 
/>
```

### Bước 3: Chiến lược "Importance"
Dựa vào mức độ quan trọng để chọn chế độ render:
- **CRITICAL**: `mode="all"` (Billboard + Tooltip)
- **IMPORTANT**: `mode="both"` (Simple + Tooltip)
- **SUPPORTING**: `mode="simple"` (Only ASCII)

---

## 📊 THÔNG SỐ TỐI ƯU (PERFORMANCE BEST PRACTICES)

| Chỉ số | Khuyến nghị | Lý do |
|--------|------------|-------|
| **Billboard Limit** | < 10 textures | Tiết kiệm VRAM, tránh bộ nhớ bị phình to. |
| **Font Atlas** | 1024x1024 px | Đủ cho 200+ ký tự bao gồm tiếng Việt và ký hiệu toán. |
| **Tooltip Trigger** | Mouse Hover | Tránh làm rối màn hình chính nhưng vẫn đầy đủ thông tin. |
| **Billboard Scale** | 2.0 | Đảm bảo chữ sắc nét khi Zoom vào Space 3D. |

---

## 💡 MẸO (PRO TIPS)
- **Blinking Cursor**: Có thể thêm hiệu ứng con trỏ nhấp nháy vào đơn thức ASCII trong Playground để tăng tính tương tác.
- **Vietnamese Support**: Luôn sử dụng hàm `removeVietnameseTones` cho layer 3D ASCII, nhưng giữ nguyên tiếng Việt có dấu cho layer Tooltip.

---

** release v1.0 - Aero-Viz Team **
