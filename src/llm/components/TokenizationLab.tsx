import React, { useState, useEffect } from 'react';

interface TokenItem {
    id: number;
    text: string;
    color: string;
}

const COLORS = ['#f97279', '#c6d8e6', '#2a72a3', '#ffc5c2', '#a0c4ff'];

export const TokenizationLab: React.FC = () => {
    const [inputText, setInputText] = useState('Dự đoán từ tiếp theo là nhiệm vụ cốt lõi của LLM.');
    const [tokens, setTokens] = useState<TokenItem[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);

    const simulateTokenize = () => {
        setIsProcessing(true);
        setTokens([]);

        // Giả lập chia từ (BPE-style)
        const words = inputText.split(' ');
        let currentTokens: TokenItem[] = [];

        words.forEach((word, idx) => {
            setTimeout(() => {
                currentTokens.push({
                    id: Math.floor(Math.random() * 50000),
                    text: word,
                    color: COLORS[idx % COLORS.length]
                });
                setTokens([...currentTokens]);
                if (idx === words.length - 1) setIsProcessing(false);
            }, idx * 200);
        });
    };

    useEffect(() => {
        simulateTokenize();
    }, []);

    return (
        <div className="tokenization-lab" style={{ padding: '30px', background: 'rgba(255, 255, 255, 0.03)', backdropFilter: 'blur(10px)', borderRadius: '16px', border: '1px solid rgba(198, 216, 230, 0.1)' }}>
            <h3 style={{ color: '#c6d8e6', fontFamily: 'Inter, sans-serif', fontWeight: '600', marginBottom: '20px' }}>🔬 Phòng Thí Nghiệm Tokenization</h3>

            <div className="input-area" style={{ marginBottom: '25px' }}>
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    style={{ background: 'rgba(0, 0, 0, 0.2)', border: '1px solid rgba(198, 216, 230, 0.2)', color: '#c6d8e6', padding: '12px 16px', width: '100%', borderRadius: '8px', fontSize: '15px' }}
                />
                <button
                    onClick={simulateTokenize}
                    disabled={isProcessing}
                    style={{ marginTop: '15px', background: '#f97279', color: '#fff', border: 'none', padding: '10px 24px', borderRadius: '8px', cursor: 'pointer', fontWeight: '600', letterSpacing: '0.5px', transition: 'all 0.3s' }}
                >
                    {isProcessing ? 'Đang mã hóa...' : 'Tokenize Now'}
                </button>
            </div>

            <div className="visual-output" style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', minHeight: '120px', alignItems: 'center' }}>
                {tokens.map((token, i) => (
                    <div
                        key={i}
                        className="token-box"
                        style={{
                            padding: '10px 16px',
                            background: i % 2 === 0 ? 'rgba(42, 114, 163, 0.15)' : 'rgba(249, 114, 121, 0.15)',
                            border: `1px solid ${token.color}`,
                            borderRadius: '10px',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
                            animation: 'popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
                        }}
                    >
                        <div style={{ fontSize: '15px', color: '#fff', fontWeight: '500' }}>"{token.text}"</div>
                        <div style={{ fontSize: '11px', color: token.color, marginTop: '6px', opacity: 0.8, letterSpacing: '1px' }}>ID: {token.id}</div>
                    </div>
                ))}
            </div>

            <style jsx>{`
                @keyframes popIn {
                    from { transform: scale(0.8); opacity: 0; }
                    to { transform: scale(1); opacity: 1; }
                }
            `}</style>
        </div>
    );
};
