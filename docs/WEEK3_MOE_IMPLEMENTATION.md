
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
# 🎯 Week 3-4 Implementation Plan: MoE Visualization Enhancement

## Current Status
✅ **MoE Detection DONE** - Code already checks `if (shape.isMoE)` at line 660+  
✅ **Basic Structure EXISTS** - Experts, gate, router already rendered  
⚠️ **Layout NEEDS IMPROVEMENT** - Currently stacks vertically, need 2x4 grid

---

## 🔄 What Exists (lines 660-760):

### ✅ Gate/Router Components:
- `gateWeight` - Router weights  
- `gateScores` - Router scores
- `gateSoftmax` - Top-K selection (softmax probabilities)

### ✅ Expert Loop:
```typescript
for (let i = 0; i < moeBlock.experts.length; i++) {
    // Creates expert weight + output
    // Currently stacks in Y direction
}
```

### ✅ Aggregation:
- `mlpResidual` - Combines expert outputs

---

## 🎨 Week 3-4 Goals:

### **Phase 1: Grid Layout (Week 3)**

#### Current (Vertical Stack):
```
Router
↓
Expert 0
Expert 1  
Expert 2
Expert 3
Expert 4
Expert 5
Expert 6
Expert 7
↓
Aggregation
```

#### Target (2x4 Grid):
```
      Router (Top)
         ↓
   ┌─────┴─────┐
   
 E0  E1  E2  E3
 E4  E5  E6  E7
 
   └─────┬─────┘
         ↓
   Aggregation
```

#### Implementation Steps:
1. **Calculate grid positions**
   ```typescript
   const expertsPerRow = 4;
   const expertRows = 2;
   const expertW = (C * cell + margin) * 1.5; // Width per expert
   const expertH = (C * cell + margin) * 2;   // Height per expert
   
   for (let i = 0; i < numExperts; i++) {
       const row = Math.floor(i / expertsPerRow);
       const col = i % expertsPerRow;
       
       const expertX = baseX + col * expertW;
       const expertY = baseY + row * expertH;
       
       // Create expert blocks at (expertX, expertY)
   }
   ```

2. **Compact expert representation**
   - Show only: Weight block + Output block per expert
   - Color code based on routing probability
   - Add expert number label

3. **Router visualization enhancement**
   - Show top-2 selection visually
   - Highlight paths to active experts
   - Display routing probabilities

---

### **Phase 2: Interactive Features (Week 4)**

#### A. Hover Tooltips
```typescript
// On expert hover:
- Expert ID
- Parameters count
- Current routing probability  
- Active/Inactive status
```

#### B. Color Coding
```typescript
// Expert colors based on selection:
const expertColor = (routingProb: number, isTopK: boolean) => {
    if (isTopK) {
        return interpolate(
            INACTIVE_COLOR,  // #888
            ACTIVE_COLOR,    // #10a37f (OpenAI green)
            routingProb
        );
    }
    return INACTIVE_COLOR;
};
```

#### C. Routing Lines
```typescript
// Draw connections from router to top-K experts
drawRoutingPath(routerOutput, expert0, routingProb0);
drawRoutingPath(routerOutput, expert1, routingProb1);
```

#### D. Expert Utilization Heatmap
```typescript
// Track which experts are used most
interface ExpertUtilization {
    expertId: number;
    usageCount: number;
    avgProbability: number;
}

// Visualize as background color intensity
```

---

## 📝 Detailed Implementation

### File: `GptModelLayout.ts`

#### Changes Needed:

**1. Grid Layout Logic (lines 696-743)**

Before:
```typescript
for (let i = 0; i < moeBlock.experts.length; i++) {
    // ...
    y += h * cell + margin;  // Stacks vertically
    // ...
    y += C * cell + margin;
}
```

After:
```typescript
const expertGrid = {
    cols: 4,
    rows: 2,
    cellW: (C * cell + margin) * 1.5,
    cellH: (C * cell + margin) * 2,
    baseX: attnLeftX - expertGrid.cols * expertGrid.cellW,
    baseY: y,
};

for (let i = 0; i < moeBlock.experts.length; i++) {
    const row = Math.floor(i / expertGrid.cols);
    const col = i % expertGrid.cols;
    
    const expertX = expertGrid.baseX + col * expertGrid.cellW;
    const expertY = expertGrid.baseY + row * expertGrid.cellH;
    
    let expFcWeight = mk({
        // ... existing code ...
        x: expertX,  // NEW: Position in grid
        y: expertY,
    });
    
    let expOut = mk({
        // ... existing code ...
        x: expertX,
        y: expertY + compact_height,
    });
}
```

**2. Router Visual Enhancement**

```typescript
// Add router block with visual connection indicator
let routerBlock = mk({
    t: 'i',
    cx: numExperts,
    cy: T,
    // Position above expert grid
    x: expertGrid.baseX + (expertGrid.cols * expertGrid.cellW) / 2,
    y: expertGrid.baseY - margin * 4,
    name: 'Router (Top-K Selection)',
    // Custom rendering for routing visualization
    special: BlkSpecial.MoERouter,
});
```

**3. Add Routing Pathway Lines**

New helper function:
```typescript
function drawExpertRouting(
    state: IProgramState,
    routerBlock: IBlkDef,
    experts: IBlkDef[],
    topKIndices: number[],
    probabilities: number[]
) {
    const routerCenter = getBlockCenter(routerBlock);
    
    experts.forEach((expert, idx) => {
        const isActive = topKIndices.includes(idx);
        const prob = probabilities[idx] || 0;
        
        if (isActive) {
            const expertCenter = getBlockCenter(expert);
            drawRoutingLine(
                state.render,
                routerCenter,
                expertCenter,
                { 
                    color: interpolateColor(INACTIVE, ACTIVE, prob),
                    thickness: 2 + prob * 2,
                    dashed: false,
                }
            );
        }
        
        // Highlight expert block
        expert.highlight = isActive ? prob : 0;
    });
}
```

---

## 🎨 Visual Design

### Color Scheme:
```typescript
const MoE_COLORS = {
    ROUTER: '#667eea',           // Blue-purple
    ACTIVE_EXPERT: '#10a37f',    // OpenAI green
    INACTIVE_EXPERT: '#6e6e80',  // Gray
    ROUTING_PATH: '#a78bfa',     // Light purple
};
```

### Expert Block Size:
- **Compact mode:** Show only weight + output (2 small blocks)
- **Detail mode:** Show full MLP structure
- **Responsive:** Adjust based on `detailLevel`

---

## 📊 Success Metrics

### Week 3:
- ✅ 2x4 expert grid renders correctly
- ✅ Router positioned above experts  
- ✅ Routing lines connect to top-2 experts
- ✅ Expert IDs visible (0-7)

### Week 4:
- ✅ Hover shows expert details
- ✅ Color intensity shows routing probability
- ✅ Animation shows token flow through experts
- ✅ Utilization heatmap tracks expert usage

---

## 🚀 Next Actions (Priority Order):

1. **Refactor expert loop to grid** (gridexpert_positions.ts snippet)
2. **Add router visual block** above grid
3. **Implement routing line rendering**
4. **Add expert labels** (E0-E7)
5. **Color coding** based on activation
6. **Interactive hover** (Week 4)
7. **Animation** (Week 4)

---

**Status:** 📋 Ready to implement  
**Estimated Time:** 6-8 hours for Week 3, 4-6 hours for Week 4  
**Current Blockers:** None  
**Dependencies:** Existing MoE detection works ✅
<!-- Aero-Footer-Start -->

## 📄 Tài liệu cùng chuyên mục
| Bài học | Liên kết |
| :--- | :--- |
| [� Kho Tài Liệu Aero-HowtoLLMs](README.md) | [Xem bài viết →](README.md) |
| [🎉 HOÀN THIỆN VISUALIZATION & CHAPTERS!](completion_visualization_and_chapters.md) | [Xem bài viết →](completion_visualization_and_chapters.md) |
| [🎉 100% LOCALIZATION COMPLETE!](localization_100_complete.md) | [Xem bài viết →](localization_100_complete.md) |
| [✅ LOCALIZATION FOUNDATION COMPLETE!](localization_summary.md) | [Xem bài viết →](localization_summary.md) |
| [✅ Việt Hóa Walkthrough - Self Attention Complete!](localization_walkthrough04.md) | [Xem bài viết →](localization_walkthrough04.md) |
| [✅ Phase 1 - Week 1: Foundation Complete!](progress_week1.md) | [Xem bài viết →](progress_week1.md) |
| [✅ Week 2 Progress: GPT-4 Integration Complete!](progress_week2.md) | [Xem bài viết →](progress_week2.md) |
| [✅ Week 3 Progress: MoE Grid Layout Complete!](progress_week3.md) | [Xem bài viết →](progress_week3.md) |
| [✅ Week 4 Complete: Router Visualization & Color Coding!](progress_week4_complete.md) | [Xem bài viết →](progress_week4_complete.md) |
| [🎯 Week 4 Progress: Router Visualization (Part 1)](progress_week4_part1.md) | [Xem bài viết →](progress_week4_part1.md) |
| [🚀 Roadmap: Mở Rộng LLM Visualization - GPT-4 & Modern Architectures](roadmap_gpt4_expansion.md) | [Xem bài viết →](roadmap_gpt4_expansion.md) |
| [🚀 Roadmap Học Hybrid AI (6 Tháng)](roadmaphybridai.md) | [Xem bài viết →](roadmaphybridai.md) |
| [🎯 LLM Training Pipeline - 3D Visualization System Design](visualization_system_design_spec.md) | [Xem bài viết →](visualization_system_design_spec.md) |
| 📌 **[🎯 Week 3-4 Implementation Plan: MoE Visualization Enhancement](week3_moe_implementation.md)** | [Xem bài viết →](week3_moe_implementation.md) |

---
## 🤝 Liên hệ & Đóng góp
Dự án được phát triển bởi **Pixibox**. Mọi đóng góp về nội dung và mã nguồn đều được chào đón.

> *"Kiến thức là để chia sẻ. Hãy cùng nhau xây dựng cộng đồng AI vững mạnh!"* 🚀

*Cập nhật tự động bởi Aero-Indexer - 2026*
<!-- Aero-Footer-End -->
