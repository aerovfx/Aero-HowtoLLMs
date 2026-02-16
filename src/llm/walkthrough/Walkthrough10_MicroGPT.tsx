import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs } from "./WalkthroughTools";

export function walkthrough10_MicroGPT({ state, layout, walkthrough: wt, tools }: IWalkthroughArgs) {
    let { c_str, c_blockRef, c_dimRef, atTime, afterTime, cleanup, breakAfter } = tools;

    // Only run when in MicroGPT phase
    if (wt.phase !== Phase.MicroGPT_Intro) {
        return;
    }


    // Introduction to MicroGPT
    commentary(wt)`
        Chào mừng đến với **MicroGPT** - một phiên bản cực kỳ đơn giản của GPT được thiết kế bởi Andrej Karpathy.
        
        MicroGPT chỉ có **4.336 tham số**, nhỏ hơn gần 20 lần so với Nano-GPT (85K tham số).
        
        Đây là mô hình lý tưởng để **hiểu rõ bản chất** của Transformer mà không bị phân tâm bởi các chi tiết phức tạp.
    `;

    commentary(wt)`
        ### Tại sao MicroGPT quan trọng?
        
        MicroGPT chứng minh rằng bạn **không cần** một mô hình khổng lồ để hiểu Transformer:
        
        - ✨ **Đơn giản**: Chỉ 1 layer, dễ debug và visualize
        - ⚡ **Nhanh**: Huấn luyện trong vài giây trên CPU
        - 🎓 **Học tập**: Mỗi component đều rõ ràng và dễ theo dõi
        - 🔬 **Thử nghiệm**: Thay đổi kiến trúc và thấy kết quả ngay lập tức
    `;

    breakAfter();

    // Architecture overview
    commentary(wt)`
        ### Kiến trúc tổng quan
        
        MicroGPT sử dụng các kỹ thuật đơn giản hóa:
        
        1. **RMSNorm** thay vì LayerNorm (không cần tính mean)
        2. **Không có bias** trong các lớp tuyến tính
        3. **Square ReLU** thay vì GELU
        4. Chỉ **1 layer Transformer** thay vì nhiều layers
        5. **Hidden dimension nhỏ**: 16 thay vì 48
        6. **Context window**: 32 tokens
    `;

    // Architecture visualization
    let t0 = afterTime(null, 1.0);
    cleanup(t0);

    commentary(wt)`
        ### Quan sát kiến trúc 3D
        
        Hãy quan sát mô hình 3D bên trái. Lưu ý rằng MicroGPT chỉ có:
        
        - 📦 **1 khối Transformer** (thay vì 3 như Nano-GPT)
        - 🔵 **RMS Norm** xuất hiện 3 lần (màu xanh cyan)
        - 🟡 **Tự chú ý đơn giản** (4 heads, mỗi head chỉ 4 dims)
        - 🟠 **MLP với Square ReLU** (màu cam)
        - ⚪ **Không có bias blocks** (so sánh với Nano-GPT)
        
        Kiến trúc tối giản này giúp bạn tập trung vào **luồng dữ liệu** thay vì chi tiết phức tạp.
    `;

    breakAfter();

    // RMSNorm explanation
    commentary(wt)`
        ### RMSNorm - Đơn giản hóa Normalization
        
        **RMSNorm** (Root Mean Square Normalization) loại bỏ bước tính mean:
        
        **LayerNorm** (phức tạp):
        \`\`\`
        mean = sum(x) / n
        variance = sum((x - mean)²) / n
        output = (x - mean) / sqrt(variance + ε)
        \`\`\`
        
        **RMSNorm** (đơn giản):
        \`\`\`
        rms = sqrt(sum(x²) / n)
        output = x / (rms + ε)
        \`\`\`
        
        💡 **Lợi ích**:
        - Giảm 50% phép tính
        - Không cần lưu mean trong backward pass
        - Vẫn ổn định training tốt!
    `;

    breakAfter();

    // No bias explanation
    commentary(wt)`
        ### Loại bỏ Bias - Giảm tham số
        
        MicroGPT **loại bỏ tất cả bias** trong các lớp Linear:
        
        **Với bias** (Nano-GPT):
        \`\`\`python
        y = x @ W + b  # b là bias vector
        \`\`\`
        
        **Không bias** (MicroGPT):
        \`\`\`python
        y = x @ W  # Chỉ có weight matrix
        \`\`\`
        
        📊 **Tác động**:
        - Giảm ~5-10% tổng số tham số
        - Đơn giản hóa tính toán
        - Vẫn đủ khả năng biểu diễn cho character-level tasks
        
        ⚠️ **Lưu ý**: Với các tác vụ phức tạp, bias vẫn quan trọng!
    `;

    breakAfter();

    // Square ReLU
    commentary(wt)`
        ### Square ReLU - Activation đơn giản
        
        **Square ReLU** thay thế GELU với công thức cực kỳ đơn giản:
        
        **GELU** (phức tạp):
        \`\`\`python
        GELU(x) = x * Φ(x)  # Φ là CDF của Gaussian
        # Hoặc xấp xỉ: x * σ(1.702 * x)
        \`\`\`
        
        **Square ReLU** (đơn giản):
        \`\`\`python
        SquareReLU(x) = (max(0, x))²
        # Hoặc: ReLU(x) * ReLU(x)
        \`\`\`
        
        📈 **Đặc điểm**:
        - Cực kỳ nhanh (chỉ 1 phép so sánh + 1 phép nhân)
        - Gradient đơn giản: 2x nếu x > 0, 0 nếu x ≤ 0
        - Hoạt động tốt cho các mô hình nhỏ
    `;

    breakAfter();

    // Comparison with Nano-GPT
    commentary(wt)`
        ### So sánh MicroGPT vs Nano-GPT
        
        | Đặc điểm | MicroGPT | Nano-GPT | Tỷ lệ |
        |----------|----------|----------|-------|
        | **Tham số** | 4.336 | 85.584 | 1:20 |
        | **Layers** | 1 | 3 | 1:3 |
        | **Hidden dim** | 16 | 48 | 1:3 |
        | **Heads** | 4 | 4 | 1:1 |
        | **Head dim** | 4 | 12 | 1:3 |
        | **Context** | 32 tokens | 11 tokens | 3:1 |
        | **Norm** | RMSNorm | LayerNorm | - |
        | **Bias** | ❌ Không | ✅ Có | - |
        | **Activation** | Square ReLU | GELU | - |
        
        🎯 **Kết luận**: MicroGPT nhỏ hơn 20 lần nhưng vẫn giữ được cấu trúc Transformer cơ bản!
    `;

    breakAfter();

    // Training characteristics
    commentary(wt)`
        ### Đặc điểm huấn luyện
        
        MicroGPT huấn luyện **cực kỳ nhanh**:
        
        ⚡ **Tốc độ**:
        - Mỗi epoch: ~2-5 giây trên CPU
        - Forward pass: ~0.1ms
        - Backward pass: ~0.2ms
        
        💾 **Bộ nhớ**:
        - Model size: ~17KB (4.336 params × 4 bytes)
        - Activations: ~2KB per batch
        - Có thể chạy trên bất kỳ thiết bị nào!
        
        📚 **Dữ liệu**:
        - Phù hợp với datasets nhỏ (vài MB)
        - Character-level tokenization
        - Có thể overfit nhanh (cần regularization)
    `;

    breakAfter();

    // Use cases
    commentary(wt)`
        ### Khi nào nên dùng MicroGPT?
        
        ✅ **Nên dùng khi**:
        - 🎓 Học cách hoạt động của Transformer
        - 🔬 Thử nghiệm kiến trúc mới (attention variants, normalization, etc.)
        - 🐛 Debug và phân tích chi tiết từng component
        - 📝 Tác vụ character-level đơn giản (tên, mã code ngắn)
        - ⚡ Cần kết quả nhanh để iterate
        - 💻 Không có GPU mạnh
        
        ❌ **Không nên dùng khi**:
        - 🏭 Production applications
        - 📖 Xử lý ngữ cảnh dài và phức tạp
        - 🌍 Multi-lingual tasks
        - 🎯 Cần độ chính xác cao
        - 📊 Datasets lớn (>100MB)
    `;

    breakAfter();

    // Code example
    commentary(wt)`
        ### Ví dụ code MicroGPT
        
        \`\`\`python
        # Định nghĩa MicroGPT config
        config = {
            'vocab_size': 32,      # Character-level
            'n_layer': 1,          # Chỉ 1 transformer block
            'n_head': 4,           # 4 attention heads
            'n_embd': 16,          # Hidden dimension
            'block_size': 32,      # Context length
            'bias': False,         # Không có bias
            'norm_type': 'rmsnorm', # RMSNorm
            'activation': 'squared_relu',
        }
        
        # Khởi tạo model
        model = MicroGPT(config)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Output: Parameters: 4,336
        \`\`\`
    `;

    breakAfter();

    // Practical tips
    commentary(wt)`
        ### Mẹo thực hành với MicroGPT
        
        🎯 **Để học tốt nhất**:
        
        1. **Bắt đầu đơn giản**: Train trên Shakespeare hoặc tên người
        2. **Visualize**: In ra attention weights và activations
        3. **Thử nghiệm**: Thay đổi từng component một
        4. **So sánh**: Đo lường tác động của mỗi thay đổi
        5. **Scale up**: Khi hiểu rõ, chuyển sang Nano-GPT
        
        🔧 **Modifications hay**:
        - Thử LayerNorm vs RMSNorm
        - So sánh ReLU, GELU, Square ReLU
        - Thêm/bớt attention heads
        - Thay đổi hidden dimension
    `;

    breakAfter();

    // Conclusion
    commentary(wt)`
        ### Kết luận
        
        MicroGPT là một **công cụ học tập tuyệt vời**!
        
        🎓 **Bài học quan trọng**:
        - Transformer có thể cực kỳ đơn giản
        - Nhiều kỹ thuật phức tạp là không cần thiết cho các tác vụ nhỏ
        - Đơn giản hóa giúp hiểu rõ bản chất của mô hình
        - "Simple is better than complex" - Zen of Python
        
        🚀 **Bước tiếp theo**:
        1. Thử nghiệm với MicroGPT
        2. Hiểu rõ từng component
        3. Chuyển sang Nano-GPT (3 layers)
        4. Cuối cùng, GPT-2 và các mô hình lớn hơn
        
        💡 **Nhớ rằng**: Mọi mô hình lớn đều bắt đầu từ những ý tưởng đơn giản như MicroGPT!
    `;

    breakAfter();
}
