import React, { useState, useEffect } from 'react';
import { getFormula } from '../../utils/latex-formula-generator';

export const LLMPlayground: React.FC = () => {
    const [temperature, setTemperature] = useState(0.7);
    const [topK, setTopK] = useState(50);
    const [topP, setTopP] = useState(0.9);
    const [prompt, setPrompt] = useState('Hệ thống LLM là...');
    const [generatedText, setGeneratedText] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);

    // Mock probabilities for visualization
    const [baseProbabilities] = useState([0.45, 0.25, 0.15, 0.1, 0.05]);
    const [modifiedProbs, setModifiedProbs] = useState(baseProbabilities);

    useEffect(() => {
        // Simple simulation of Temperature effect on probability distribution
        const logits = baseProbabilities.map(p => Math.log(p));
        const tempLogits = logits.map(l => l / temperature);
        const expLogits = tempLogits.map(l => Math.exp(l));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const newProbs = expLogits.map(p => p / sumExp);
        setModifiedProbs(newProbs);
    }, [temperature, baseProbabilities]);

    const handleGenerate = () => {
        setIsGenerating(true);
        setGeneratedText('');
        let currentText = '';
        const words = ['một', 'kiến', 'trúc', 'đột', 'phá', 'trong', 'xử', 'lý', 'ngôn', 'ngữ', 'tự', 'nhiên', 'hiện', 'nay.'];
        let wordIndex = 0;

        const interval = setInterval(() => {
            if (wordIndex < words.length) {
                currentText += words[wordIndex] + ' ';
                setGeneratedText(currentText);
                wordIndex++;
            } else {
                clearInterval(interval);
                setIsGenerating(false);
            }
        }, 300);
    };

    return (
        <div className="llm-playground" style={{
            padding: '30px',
            background: 'rgba(255, 255, 255, 0.03)',
            backdropFilter: 'blur(20px)',
            borderRadius: '24px',
            border: '1px solid rgba(198, 216, 230, 0.1)',
            color: '#c6d8e6'
        }}>
            <h3 style={{ marginBottom: '25px', color: '#f97279' }}>🎮 Level 5: Interactive Playground</h3>

            <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '30px' }}>
                {/* Control Panel */}
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '16px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <h4 style={{ fontSize: '14px', borderBottom: '1px solid rgba(198, 216, 230, 0.1)', paddingBottom: '10px' }}>CẤU HÌNH THAM SỐ</h4>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
                            <span>Temperature: <strong style={{ color: '#f97279' }}>{temperature}</strong></span>
                        </div>
                        <input
                            type="range" min="0.1" max="2.0" step="0.1"
                            value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            style={{ width: '100%', accentColor: '#f97279' }}
                        />
                        <p style={{ fontSize: '10px', opacity: 0.6, marginTop: '4px' }}>T thấp = Hội tụ, T cao = Sáng tạo/Ngẫu nhiên</p>
                    </div>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
                            <span>Top-K: <strong style={{ color: '#2a72a3' }}>{topK}</strong></span>
                        </div>
                        <input
                            type="range" min="1" max="100" step="1"
                            value={topK} onChange={(e) => setTopK(parseInt(e.target.value))}
                            style={{ width: '100%', accentColor: '#2a72a3' }}
                        />
                    </div>

                    <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
                            <span>Top-P (Nucleus): <strong style={{ color: '#c6d8e6' }}>{topP}</strong></span>
                        </div>
                        <input
                            type="range" min="0.1" max="1.0" step="0.05"
                            value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))}
                            style={{ width: '100%', accentColor: '#c6d8e6' }}
                        />
                    </div>

                    <div style={{ marginTop: '10px', padding: '15px', background: 'rgba(42, 114, 163, 0.1)', borderRadius: '12px' }}>
                        <span style={{ fontSize: '11px', fontWeight: 'bold' }}>Biểu đồ xác suất (Softmax Distribution)</span>
                        <div style={{ display: 'flex', alignItems: 'flex-bottom', gap: '4px', height: '60px', marginTop: '10px' }}>
                            {modifiedProbs.map((p, i) => (
                                <div key={i} style={{
                                    flex: 1,
                                    height: `${p * 100}%`,
                                    background: i === 0 ? '#f97279' : '#2a72a3',
                                    borderRadius: '2px 2px 0 0',
                                    transition: 'height 0.3s'
                                }}></div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Simulation Area */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Nhập prompt của bạn..."
                            style={{
                                flex: 1,
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(198, 216, 230, 0.2)',
                                padding: '12px 20px',
                                borderRadius: '12px',
                                color: '#fff',
                                outline: 'none'
                            }}
                        />
                        <button
                            onClick={handleGenerate}
                            disabled={isGenerating}
                            style={{
                                padding: '0 25px',
                                borderRadius: '12px',
                                border: 'none',
                                background: isGenerating ? '#333' : '#f97279',
                                color: '#fff',
                                fontWeight: 'bold',
                                cursor: 'pointer'
                            }}
                        >
                            {isGenerating ? 'Đang chạy...' : 'RUN'}
                        </button>
                    </div>

                    <div style={{
                        flex: 1,
                        background: 'rgba(0,0,0,0.4)',
                        borderRadius: '16px',
                        padding: '25px',
                        fontSize: '18px',
                        lineHeight: '1.8',
                        border: '1px solid rgba(42, 114, 163, 0.2)',
                        minHeight: '200px',
                        fontFamily: 'Roboto Mono, monospace'
                    }}>
                        <span style={{ color: '#f97279' }}>{prompt}</span> {generatedText}
                        {isGenerating && <span className="cursor-blink">|</span>}
                    </div>

                    <div style={{ display: 'flex', gap: '15px' }}>
                        <div style={{ padding: '10px 15px', background: 'rgba(42, 114, 163, 0.1)', borderRadius: '8px', border: '1px solid #2a72a344', fontSize: '11px' }}>
                            🎯 <strong>Token Output:</strong> {Math.floor(Math.random() * 50) + 120} tokens/sec
                        </div>
                        <div style={{ padding: '10px 15px', background: 'rgba(249, 114, 121, 0.1)', borderRadius: '8px', border: '1px solid #f9727944', fontSize: '11px' }}>
                            🔋 <strong>VRAM:</strong> 12.4 GB used
                        </div>
                    </div>
                </div>
            </div>

            <style jsx>{`
                .cursor-blink {
                    animation: blink 0.8s infinite;
                    font-weight: bold;
                    color: #f97279;
                }
                @keyframes blink {
                    50% { opacity: 0; }
                }
                input[type=range] {
                    cursor: pointer;
                }
            `}</style>
        </div>
    );
};
