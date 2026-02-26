
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
# ✅ Week 3 Progress: MoE Grid Layout Complete!

## 📦 Changes Implemented

### **File: `GptModelLayout.ts` (lines 699-747)**

#### Before (Vertical Stack):
Expert 0  ←─ y position
  ↓ h*cell

$$
Expert 1  ←─ y += height
$$

  ↓
... (stacked vertically, very tall)

#### After (2x4 Grid):
        E0    E1    E2    E3
        E4    E5    E6    E7
    ←─────────────────────→
       grid positioning

---

## 🎨 Technical Implementation

### **Grid Calculation Logic:**

```typescript
// Expert Grid Parameters

$$
const expertsPerRow = 4;                    // 4 experts per row
$$

$$
const row = Math.floor(i / expertsPerRow);  // Row: 0 or 1
$$

$$

$$

const col = i % expertsPerRow;              // Col: 0,1,2,3

$$

$$

// Compact sizing

$$

$$

const expertW = C * cell * 0.8;             // 80% of original width

$$

$$

const expertH = (h + C) * cell + margin * 2; // Height for weight + output

$$

$$

const spacingX = expertW + margin * 3;      //  Horizontal spacing

$$

$$

$$
const spacingY = expertH + margin * 2;      // Vertical spacing
$$

$$
// Grid base position (to the left of attention layers)
$$

$$
const gridBaseX = attnLeftX - (expertsPerRow * spacingX);
$$

$$

$$

const gridBaseY = y + row * spacingY;

$$

$$

// Individual expert position

$$

$$

const expertX = gridBaseX + col * spacingX;

$$

$$

$$
const expertY = gridBaseY;
$$

$$
### **Per-Expert Visualization:** Each expert now renders: 1. **Weight Block** (`expFcWeight`) - Expert's weight matrix 2. **Output Block** (`expOut`) - Expert's output **Compact representation:** - Labels: "Chuyên gia 0 W1" → "Chuyên gia 7 W1" - Output labels: "Chuyên gia 0 Đầu ra" → "E7 Đầu ra" - Both blocks positioned in grid layout --- ## 📊 Visual Improvements ### Space Efficiency: | Layout | Height (model units) | Width Utilization | |--------|---------------------|-------------------| | **Before (Stack)** | ~8 × expertH | Narrow (1 column) | | **After (Grid)** | ~2 × expertH | Wide (4 columns) ✅ | **Space saved:** ~75% vertical space! ### Positioning Strategy: Attention Layers (attnLeftX) │ ├──────────── margin ────────────┤ │                                  │ [Router]                               │ │                                  │ ┌─────┴─────┐                           │ │           │                           │ E0  E1  E2  E3 ←── gridBaseX             │ E4  E5  E6  E7                            │ │                                       │ └───────────┬───────────────────────────┘
$$

Aggregation

$$
--- ## 🔍 Code Changes Detail ### **Key Modifications:** **1. Removed vertical Y increment** ```typescript // BEFORE:
$$

y += h * cell + margin;  // After weight block

$$

$$

y += C * cell + margin;  // After output block

$$
// AFTER: // No individual Y increments! // All positioning relative to grid coordinates **2. Added grid position calculation** ```typescript // NEW: Grid math
$$

$$
const row = Math.floor(i / 4);
$$

$$

$$

const col = i % 4;

$$

$$

$$
const expertX = gridBaseX + col * spacingX;
$$

$$

$$

const expertY = gridBaseY + row * spacingY;

$$

$$

**3. Updated final Y position**
```typescript
// BEFORE: (implicitly at end of last expert)

// AFTER:

$$
y += Math.ceil(moeBlock.experts.length / 4) * (C * cell * 2 + margin * 4);
$$

// Jumps to after entire grid (2 rows)

**4. Changed positioning parameters**
```typescript
// BEFORE:
xR: attnLeftX,  // Right-aligned to attention
xL: attnLeftX + margin,  // Left from attention
y: y,  // stacking vertically

// AFTER:
xL: expertX,  // Absolute grid position
y: expertY,   // Grid row position

---

## ✅ Success Metrics (Week 3)

| Goal | Status | Details |
|------|--------|---------|
| 2x4 expert grid | ✅ DONE | 8 experts in 2 rows, 4 columns |
| Positioned to left | ✅ DONE | `gridBaseX = attnLeftX - (4 * spacingX)` |
| Compact visualization | ✅ DONE | 80% width, simplified blocks |
| TypeScript compiles | ✅ DONE | Fixed `x` → `xL` property |
| Dev server runs | ✅ DONE | No runtime errors |
| Expert labels visible | ✅ DONE | "Chuyên gia 0-7" |

---

## 🎯 Next Steps (Week 4)

### **Still TODO:**

1. **Router Visualization** ⏳
   - Add router block above grid
   - Show top-K selection visually
   - Routing probability display

2. **Routing Lines** ⏳
   - Draw connections from router to active experts
   - Line thickness = routing probability
   - Color: active vs inactive

3. **Interactive Features** ⏳
   - Hover tooltips on experts
   - Show expert details (ID, params, activation)
   - Click to highlight routing path

4. **Color Coding** ⏳
   ```typescript

$$

$$

expert.highlight = isTopK ? routingProb : 0;

$$

$$

$$
expert.color = isTopK ? ACTIVE_GREEN : INACTIVE_GRAY;
$$
