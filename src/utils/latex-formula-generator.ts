/**
 * LaTeX Formula Generator for LLM Architecture
 * Generates LaTeX code for common formulas in transformer models
 */

export interface LatexFormula {
    name: string;
    latex: string;
    description: string;
    asciiMath?: string; // ASCII fallback for 3D rendering
}

export const LLM_FORMULAS: Record<string, LatexFormula> = {
    // ========== ATTENTION MECHANISM ==========

    ATTENTION_BASIC: {
        name: "Basic Attention",
        latex: String.raw`\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V`,
        asciiMath: "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
        description: "Cơ chế attention cơ bản"
    },

    MULTI_HEAD_ATTENTION: {
        name: "Multi-Head Attention",
        latex: String.raw`\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}`,
        asciiMath: "MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O",
        description: "Multi-head attention với nhiều attention heads"
    },

    SCALED_DOT_PRODUCT: {
        name: "Scaled Dot-Product Attention",
        latex: String.raw`\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V`,
        asciiMath: "score = (Q * K^T) / sqrt(d_k)",
        description: "Tính attention scores với scaling factor"
    },

    ATTENTION_SCORES: {
        name: "Attention Scores",
        latex: String.raw`e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}`,
        asciiMath: "e_ij = (q_i · k_j) / sqrt(d_k)",
        description: "Điểm attention giữa query i và key j"
    },

    ATTENTION_WEIGHTS: {
        name: "Attention Weights",
        latex: String.raw`\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})}`,
        asciiMath: "alpha_ij = exp(e_ij) / sum(exp(e_ik))",
        description: "Trọng số attention sau softmax"
    },

    // ========== POSITIONAL ENCODING ==========

    POSITIONAL_ENCODING_SIN: {
        name: "Positional Encoding (Sin)",
        latex: String.raw`PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)`,
        asciiMath: "PE(pos,2i) = sin(pos / 10000^(2i/d_model))",
        description: "Positional encoding với sin cho vị trí chẵn"
    },

    POSITIONAL_ENCODING_COS: {
        name: "Positional Encoding (Cos)",
        latex: String.raw`PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)`,
        asciiMath: "PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))",
        description: "Positional encoding với cos cho vị trí lẻ"
    },

    // ========== FEED-FORWARD NETWORK ==========

    FFN: {
        name: "Feed-Forward Network",
        latex: String.raw`\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2`,
        asciiMath: "FFN(x) = max(0, xW_1 + b_1)W_2 + b_2",
        description: "Feed-forward network với ReLU activation"
    },

    FFN_GELU: {
        name: "FFN with GELU",
        latex: String.raw`\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2`,
        asciiMath: "FFN(x) = GELU(xW_1 + b_1)W_2 + b_2",
        description: "Feed-forward network với GELU activation"
    },

    GELU_ACTIVATION: {
        name: "GELU Activation",
        latex: String.raw`\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]`,
        asciiMath: "GELU(x) = x * Phi(x)",
        description: "Gaussian Error Linear Unit activation"
    },

    // ========== LAYER NORMALIZATION ==========

    LAYER_NORM: {
        name: "Layer Normalization",
        latex: String.raw`\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta`,
        asciiMath: "LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta",
        description: "Layer normalization"
    },

    LAYER_NORM_MEAN: {
        name: "Layer Norm Mean",
        latex: String.raw`\mu = \frac{1}{d}\sum_{i=1}^{d} x_i`,
        asciiMath: "mu = (1/d) * sum(x_i)",
        description: "Giá trị trung bình trong layer norm"
    },

    LAYER_NORM_VAR: {
        name: "Layer Norm Variance",
        latex: String.raw`\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2`,
        asciiMath: "sigma^2 = (1/d) * sum((x_i - mu)^2)",
        description: "Phương sai trong layer norm"
    },

    // ========== RESIDUAL CONNECTION ==========

    RESIDUAL: {
        name: "Residual Connection",
        latex: String.raw`\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))`,
        asciiMath: "Output = LayerNorm(x + Sublayer(x))",
        description: "Kết nối residual với layer normalization"
    },

    // ========== EMBEDDINGS ==========

    TOKEN_EMBEDDING: {
        name: "Token Embedding",
        latex: String.raw`E = \text{Embedding}(x) \in \mathbb{R}^{d_{model}}`,
        asciiMath: "E = Embedding(x)",
        description: "Vector embedding của token"
    },

    EMBEDDING_WITH_POS: {
        name: "Embedding with Position",
        latex: String.raw`x = \text{TokenEmbed}(w) + \text{PosEmbed}(pos)`,
        asciiMath: "x = TokenEmbed(w) + PosEmbed(pos)",
        description: "Token embedding + positional encoding"
    },

    // ========== SOFTMAX ==========

    SOFTMAX: {
        name: "Softmax",
        latex: String.raw`\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}`,
        asciiMath: "softmax(x_i) = exp(x_i) / sum(exp(x_j))",
        description: "Hàm softmax cho probability distribution"
    },

    // ========== CROSS ENTROPY LOSS ==========

    CROSS_ENTROPY: {
        name: "Cross Entropy Loss",
        latex: String.raw`\mathcal{L} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)`,
        asciiMath: "L = -sum(y_i * log(y_hat_i))",
        description: "Cross entropy loss function"
    },

    // ========== PERPLEXITY ==========

    PERPLEXITY: {
        name: "Perplexity",
        latex: String.raw`\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{<i})\right)`,
        asciiMath: "PPL = exp(-(1/N) * sum(log P(w_i | w_<i)))",
        description: "Perplexity metric đánh giá language model"
    },

    // ========== TRANSFORMER ARCHITECTURE ==========

    TRANSFORMER_ENCODER: {
        name: "Transformer Encoder",
        latex: String.raw`\begin{aligned}
z_0 &= x + \text{PE} \\
z_\ell &= \text{TransformerBlock}_\ell(z_{\ell-1}), \quad \ell = 1, ..., L \\
\text{Output} &= z_L
\end{aligned}`,
        asciiMath: "z_0 = x + PE; z_l = TransformerBlock_l(z_{l-1})",
        description: "Kiến trúc Transformer Encoder"
    },

    TRANSFORMER_DECODER: {
        name: "Transformer Decoder",
        latex: String.raw`\begin{aligned}
z_0 &= y + \text{PE} \\
z_\ell &= \text{DecoderBlock}_\ell(z_{\ell-1}, \text{encoder\_output}), \quad \ell = 1, ..., L \\
\text{Output} &= \text{softmax}(z_L W + b)
\end{aligned}`,
        asciiMath: "z_0 = y + PE; z_l = DecoderBlock_l(z_{l-1}, enc_out)",
        description: "Kiến trúc Transformer Decoder"
    },

    // ========== GPT-STYLE (CAUSAL) ==========

    CAUSAL_MASK: {
        name: "Causal Attention Mask",
        latex: String.raw`\text{Mask}_{ij} = \begin{cases} 
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}`,
        asciiMath: "Mask_ij = 0 if i>=j, else -inf",
        description: "Causal mask để ngăn attention vào future tokens"
    },

    // ========== BERT-STYLE (MASKED LM) ==========

    MASKED_LM_OBJECTIVE: {
        name: "Masked LM Objective",
        latex: String.raw`\mathcal{L}_{\text{MLM}} = -\mathbb{E}\left[\log P(x_i | \mathbf{x}_{\backslash i})\right]`,
        asciiMath: "L_MLM = -E[log P(x_i | x_{\\i})]",
        description: "Masked Language Modeling objective"
    },

    // ========== SCALING LAWS ==========

    SCALING_LAW: {
        name: "Neural Scaling Law",
        latex: String.raw`L(N) = \left(\frac{N_c}{N}\right)^\alpha`,
        asciiMath: "L(N) = (N_c / N)^alpha",
        description: "Scaling law: loss theo số parameters"
    },

    COMPUTE_OPTIMAL: {
        name: "Compute-Optimal Scaling",
        latex: String.raw`N_{\text{opt}} \propto C^a, \quad D_{\text{opt}} \propto C^b`,
        asciiMath: "N_opt ~ C^a, D_opt ~ C^b",
        description: "Scaling tối ưu cho compute, data và model size"
    },

    // ========== RLHF ==========

    REWARD_MODEL: {
        name: "Reward Model",
        latex: String.raw`r_\theta(x, y) = \text{Reward}(\text{Prompt: } x, \text{Response: } y)`,
        asciiMath: "r(x,y) = Reward(prompt: x, response: y)",
        description: "Reward model trong RLHF"
    },

    PPO_OBJECTIVE: {
        name: "PPO Objective",
        latex: String.raw`\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]`,
        asciiMath: "L_PPO = E[min(r_t * A_hat, clip(r_t) * A_hat)]",
        description: "PPO objective function trong RLHF"
    },

    // ========== KL DIVERGENCE ==========

    KL_DIVERGENCE: {
        name: "KL Divergence",
        latex: String.raw`D_{KL}(P \| Q) = \sum_{x} P(x) \log\frac{P(x)}{Q(x)}`,
        asciiMath: "D_KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))",
        description: "KL divergence để đo sự khác biệt giữa distributions"
    },

    // ========== TEMPERATURE SAMPLING ==========

    TEMPERATURE_SOFTMAX: {
        name: "Temperature Sampling",
        latex: String.raw`P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}`,
        asciiMath: "P(x_i) = exp(z_i/T) / sum(exp(z_j/T))",
        description: "Softmax với temperature T để control randomness"
    },

    // ========== TOP-K / TOP-P SAMPLING ==========

    TOP_K_SAMPLING: {
        name: "Top-K Sampling",
        latex: String.raw`P_K(x) = \begin{cases}
\frac{P(x)}{\sum_{x' \in V_K} P(x')} & \text{if } x \in V_K \\
0 & \text{otherwise}
\end{cases}`,
        asciiMath: "Sample from top-K highest probability tokens",
        description: "Chỉ sample từ K tokens có xác suất cao nhất"
    },

    TOP_P_SAMPLING: {
        name: "Top-P (Nucleus) Sampling",
        latex: String.raw`V_P = \text{smallest set such that } \sum_{x \in V_P} P(x) \geq p`,
        asciiMath: "V_P = smallest set where sum(P(x)) >= p",
        description: "Sample từ smallest set có tổng xác suất >= p"
    },

    // ========== GRADIENT COMPUTATION ==========

    BACKPROP: {
        name: "Backpropagation",
        latex: String.raw`\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial W}`,
        asciiMath: "dL/dW = (dL/dy) * (dy/dW)",
        description: "Chain rule trong backpropagation"
    },

    ADAM_UPDATE: {
        name: "Adam Optimizer",
        latex: String.raw`\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\theta_t &= \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}`,
        asciiMath: "theta = theta - alpha * m / (sqrt(v) + eps)",
        description: "Adam optimizer update rule"
    },

    LEARNING_RATE_WARMUP: {
        name: "Learning Rate Warmup",
        latex: String.raw`\eta_t = \eta_{\max} \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)`,
        asciiMath: "lr_t = lr_max * min(1, t/T_warmup)",
        description: "Linear warmup cho learning rate"
    },
};

/**
 * Convert LaTeX to ASCII-safe format for 3D rendering
 */
export function latexToAscii(latex: string): string {
    return latex
        .replace(/\\text\{([^}]+)\}/g, '$1')
        .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)')
        .replace(/\\sqrt\{([^}]+)\}/g, 'sqrt($1)')
        .replace(/\\sum_\{([^}]+)\}\^\{([^}]+)\}/g, 'sum_$1^$2')
        .replace(/\\exp/g, 'exp')
        .replace(/\\log/g, 'log')
        .replace(/\\mathbb\{R\}/g, 'R')
        .replace(/\\odot/g, '*')
        .replace(/\^T/g, '^T')
        .replace(/\\_/g, '_')
        .replace(/\\\\/g, ' ')
        .replace(/[{}]/g, '');
}

/**
 * Get formula by name
 */
export function getFormula(name: keyof typeof LLM_FORMULAS): LatexFormula {
    return LLM_FORMULAS[name];
}

/**
 * Get all formulas in a category
 */
export function getFormulasByCategory(category: string): LatexFormula[] {
    const categories: Record<string, (keyof typeof LLM_FORMULAS)[]> = {
        'attention': [
            'ATTENTION_BASIC',
            'MULTI_HEAD_ATTENTION',
            'SCALED_DOT_PRODUCT',
            'ATTENTION_SCORES',
            'ATTENTION_WEIGHTS',
        ],
        'position': [
            'POSITIONAL_ENCODING_SIN',
            'POSITIONAL_ENCODING_COS',
        ],
        'ffn': [
            'FFN',
            'FFN_GELU',
            'GELU_ACTIVATION',
        ],
        'normalization': [
            'LAYER_NORM',
            'LAYER_NORM_MEAN',
            'LAYER_NORM_VAR',
        ],
        'embedding': [
            'TOKEN_EMBEDDING',
            'EMBEDDING_WITH_POS',
        ],
        'loss': [
            'CROSS_ENTROPY',
            'PERPLEXITY',
        ],
        'sampling': [
            'SOFTMAX',
            'TEMPERATURE_SOFTMAX',
            'TOP_K_SAMPLING',
            'TOP_P_SAMPLING',
        ],
        'rlhf': [
            'REWARD_MODEL',
            'PPO_OBJECTIVE',
            'KL_DIVERGENCE',
        ],
        'training': [
            'BACKPROP',
            'ADAM_UPDATE',
            'LEARNING_RATE_WARMUP',
        ],
    };

    const keys = categories[category] || [];
    return keys.map(key => LLM_FORMULAS[key]);
}

/**
 * Generate LaTeX document for all formulas
 */
export function generateLatexDocument(): string {
    const header = String.raw`\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\title{LLM Architecture Formulas}
\author{Generated}
\date{\today}

\begin{document}
\maketitle

`;

    const sections = [
        { title: 'Attention Mechanism', category: 'attention' },
        { title: 'Positional Encoding', category: 'position' },
        { title: 'Feed-Forward Networks', category: 'ffn' },
        { title: 'Normalization', category: 'normalization' },
        { title: 'Embeddings', category: 'embedding' },
        { title: 'Loss Functions', category: 'loss' },
        { title: 'Sampling Methods', category: 'sampling' },
        { title: 'RLHF', category: 'rlhf' },
        { title: 'Training', category: 'training' },
    ];

    let content = '';

    sections.forEach(({ title, category }) => {
        content += `\\section{${title}}\n\n`;
        const formulas = getFormulasByCategory(category);

        formulas.forEach(formula => {
            content += `\\subsection{${formula.name}}\n`;
            content += `${formula.description}\n\n`;
            content += `\\begin{equation}\n${formula.latex}\n\\end{equation}\n\n`;
        });
    });

    const footer = String.raw`\end{document}`;

    return header + content + footer;
}

// Export for use in visualization
export default LLM_FORMULAS;
