
<!-- Aero-Navigation-Start -->
**Home**

---
### 🧭 Điều hướng nhanh

- [🏠 Cổng tài liệu](index.md)
- [📚 Module 01: LLM Course](01_llm_course/index.md)
- [🔢 Module 02: Tokenization](02_words_to_tokens_to_numbers/index.md)
- [🏗️ Module 04: Build GPT](04_buildgpt/index.md)
- [🎯 Module 07: Fine-tuning](07_fine_tune_pretrained_models/index.md)
- [🔍 Module 19: AI Safety](19_ai_safety/index.md)
- [🐍 Module 20: Python for AI](20_python_colab_notebooks/index.md)
---
<!-- Aero-Navigation-End -->
# 🎯 Week 4 Progress: Router Visualization (Part 1)

## ✅ Completed So Far

### **1. Router Repositioning** ✅

#### Before:
Router blocks (gateWeight, gateScores, gateSoftmax)
  ↓ (stacked right-aligned)
Experts grid

#### After:
       Router (CENTERED)
           ↓
        ┌──┴──┐
      E0 E1 E2 E3
      E4 E5 E6 E7

**Key Changes:**
- Router components now use `xM` $Middle/Center$ instead of `xR` (Right-aligned)
- Position calculated as `gridCenterX = attnLeftX - gridWidth / 2 - margin * 4`
- All router blocks (`gateWeight`, `gateScores`, `gateSoftmax`) centered

### **2. Router Labels Enhanced** ✅

**Before:**
- "Trọng số Cổng"
- "Điểm số Cổng"  
- "Xác suất Cổng"

**After:**
- "Router: Trọng số"
- "Router: Điểm số"
- "Router: Top-K"

More intuitive naming that shows these are part of the routing mechanism!

### **3. Grid-Aware Positioning** ✅

Router now pre-calculates expert grid dimensions:
```typescript
const expertsPerRow = 4;
const expertW = C * cell * 0.8;
const expertSpacingX = expertW + margin * 3;
const gridWidth = expertsPerRow * expertSpacingX;
const gridCenterX = attnLeftX - gridWidth / 2 - margin * 4;

This ensures router is ALWAYS centered above the expert grid regardless of expert sizes!

---

## 📊 Visual Layout Now

$$
Input: ln2.lnResid
$$

↓
        ┌─────────┴─────────┐
        │                    │
    [Router: Trọng số]      │
         (Gate Weight)       │
              ↓              │
    [Router: Điểm số]       │
       (Routing Logits)     │
              ↓              │
    [Router: Top-K]         │
      (Softmax Probs)       │
              ↓              │
        ┌────┴────┐         │
        │         │         │
      E0 E1 E2 E3 │         │
      E4 E5 E6 E7 │         │
        │         │         │
        └────┬────┘         │
             ↓              │
    [MoE (Cộng dồn)]       │
      (Aggregation)────────┘
             ↓

$$
Output
$$

---

## 🎨 Next Steps (Part 2 - In Progress)

### **Still TODO:**

1. **Color Coding for Experts** ⏳
   ```typescript
   // Pseudo-code:
   if (expertIsInTopK(i)) {
       expFcWeight.highlight = routingProbability[i];
       expOut.highlight = routingProbability[i];
   } else {
       expFcWeight.opacity = 0.3;  // Inactive/dimmed
       expOut.opacity = 0.3;
   }

2. **Routing Line Visualization** ⏳
   - Draw lines from `gateSoftmax` to top-K experts
   - Line thickness ∝ routing probability
   - Color gradient (inactive gray → active green)

3. **Interactive Hover** (Later)
   - Tooltip showing expert details
   - Highlight routing path on hover

---

## 🔧 Implementation Details

### Code Changes:

**File: `GptModelLayout.ts`**

#### Lines 652-662: Router Weight
```typescript
let gateWeight = mk({
    t: 'w', cx: C, cz: 1, cy: numExperts, y: y,
    xM: gridCenterX, zM: 0,  // ← CENTERED
    // ...
    name: 'Router: Trọng số',  // ← NEW LABEL
    small: true,  // ← Compact display
});

#### Lines 674-685: Router Scores
```typescript
let gateScores = mk({
    t: 'i', cx: T, cz: B, cy: numExperts, y: y,
    xM: gridCenterX, zM: 0,  // ← CENTERED
    // ...
    name: 'Router: Điểm số',
    small: true,
});

#### Lines 687-697: Router Softmax (Top-K)
```typescript
let gateSoftmax = mk({
    t: 'i', cx: T, cz: B, cy: numExperts, y: y,
    xM: gridCenterX, zM: 0,  // ← CENTERED
    // ...
    name: 'Router: Top-K',
});

---

## 📈 Progress Metrics

| Feature | Status | Complexity | Time |
|---------|--------|------------|------|
| Router centering | ✅ DONE | Medium | 1h |
| Enhanced labels | ✅ DONE | Low | 15min |
| Grid-aware positioning | ✅ DONE | Medium | 30min |
| **Part 1 Total** | **✅** | **-** | **~1.75h** |
| | | | |
| Color coding | ⏳ TODO | Medium | ~1h |
| Routing lines | ⏳ TODO | High | ~2h |
| Interactive hover | ⏳ TODO | Medium | ~1.5h |
| **Part 2 Estimate** | **⏳** | **-** | **~4.5h** |

---

## 🎯 Current Visual State

### Router Position:
- ✅ Centered horizontally above expert grid
- ✅ Proper vertical spacing
- ✅ Consistent with expert grid layout

### Expert Grid:
- ✅ 2 rows × 4 columns
- ✅ Positioned to left of attention
- ✅ Compact visualization

### Routing Mechanism:
- ✅ Router exists and processes
- ✅ Softmax for top-K selection
- ⏳ Visual connections (TODO)
- ⏳ Color coding (TODO)

---

## 🐛 Known Issues

**None! ✅**
- TypeScript compiles successfully
- Router positioning works correctly
- No runtime errors

---

## 💡 Design Decisions

### Why Center Router?
- **Visual hierarchy:** Router is the decision maker → centered position emphasizes importance
- **Flow clarity:** Clear top-down flow: Input → Router → Experts → Aggregation
- **Symmetry:** Centered router above symmetric 2x4 grid creates balanced visual

### Why Small Router Blocks?
- **Focus on experts:** Experts are the compute-heavy part
- **Reduced clutter:** Smaller router blocks don't dominate the view
- **Detail levels:** Can expand when `detailLevel >= 2`

### Label Changes:
- **"Router:"** prefix makes role crystal clear
- **"Top-K"** more intuitive than "Xác suất Cổng" for routing concept

---

## 🚀 Next Implementation Steps

### **Immediate (Part 2):**

1. **Add color coding logic**
   ```typescript
   // In expert loop:
   const isActive = i < shape.expertsActive;  // Top-2 for GPT-4
   
   expFcWeight.highlight = isActive ? 0.5 : 0;
   expFcWeight.opacity = isActive ? 1.0 : 0.4;
   
   expOut.highlight = isActive ? 0.5 : 0;
   expOut.opacity = isActive ? 1.0 : 0.4;

2. **Create routing line helper function**
   ```typescript
   function drawRoutingLines(
       router: IBlkDef,
       experts: IBlkDef[],
       topKIndices: number[]
   ) {
       // Draw lines from router to top-K experts
   }

3. **Integrate into render pipeline**
   - Call after expert rendering
   - Use existing line drawing utilities

---

**Status:** 🟡 Week 4 Part 1 Complete (Router Positioning ✅)  
**Next:** Part 2 - Color Coding & Routing Lines  
**ETA Part 2:** ~2-3 hours  
**Overall Week 4:** ~50% complete

---

**Date:** 2026-02-15  
**Phase:** 1.2 - MoE Visualization $Week 4/4$  
**Sub-task:** Router Visualization Part 1/2
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [🎉 HOÀN THIỆN VISUALIZATION & CHAPTERS!](COMPLETION_VISUALIZATION_AND_CHAPTERS.md) | [Xem bài viết →](COMPLETION_VISUALIZATION_AND_CHAPTERS.md) |
| [🎉 100% LOCALIZATION COMPLETE!](LOCALIZATION_100_COMPLETE.md) | [Xem bài viết →](LOCALIZATION_100_COMPLETE.md) |
| [✅ LOCALIZATION FOUNDATION COMPLETE!](LOCALIZATION_SUMMARY.md) | [Xem bài viết →](LOCALIZATION_SUMMARY.md) |
| [✅ Việt Hóa Walkthrough - Self Attention Complete!](LOCALIZATION_WALKTHROUGH04.md) | [Xem bài viết →](LOCALIZATION_WALKTHROUGH04.md) |
| [✅ Phase 1 - Week 1: Foundation Complete!](PROGRESS_WEEK1.md) | [Xem bài viết →](PROGRESS_WEEK1.md) |
| [✅ Week 2 Progress: GPT-4 Integration Complete!](PROGRESS_WEEK2.md) | [Xem bài viết →](PROGRESS_WEEK2.md) |
| [✅ Week 3 Progress: MoE Grid Layout Complete!](PROGRESS_WEEK3.md) | [Xem bài viết →](PROGRESS_WEEK3.md) |
| [✅ Week 4 Complete: Router Visualization & Color Coding!](PROGRESS_WEEK4_COMPLETE.md) | [Xem bài viết →](PROGRESS_WEEK4_COMPLETE.md) |
| 📌 **[🎯 Week 4 Progress: Router Visualization (Part 1)](PROGRESS_WEEK4_PART1.md)** | [Xem bài viết →](PROGRESS_WEEK4_PART1.md) |
| [� Kho Tài Liệu Aero-HowtoLLMs](README.md) | [Xem bài viết →](README.md) |
| [🚀 Roadmap: Mở Rộng LLM Visualization - GPT-4 & Modern Architectures](ROADMAP_GPT4_EXPANSION.md) | [Xem bài viết →](ROADMAP_GPT4_EXPANSION.md) |
| [🎯 LLM Training Pipeline - 3D Visualization System Design](VISUALIZATION_SYSTEM_DESIGN_SPEC.md) | [Xem bài viết →](VISUALIZATION_SYSTEM_DESIGN_SPEC.md) |
| [🎯 Week 3-4 Implementation Plan: MoE Visualization Enhancement](WEEK3_MOE_IMPLEMENTATION.md) | [Xem bài viết →](WEEK3_MOE_IMPLEMENTATION.md) |
| [🚀 Roadmap Học Hybrid AI (6 Tháng)](roadmapHybridAI.md) | [Xem bài viết →](roadmapHybridAI.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
