import React, { useState, useEffect } from 'react';
import { getFormula } from '../../utils/latex-formula-generator';

export const AttentionMicroscope: React.FC = () => {
    const [step, setStep] = useState(0);
    const formula = getFormula('ATTENTION_BASIC');

    // Mock data for visualization
    const qVector = [0.2, 0.8, -0.1, 0.5];
    const kVectors = [
        [0.3, 0.7, 0.1, 0.4],
        [0.1, 0.2, 0.9, -0.1],
        [0.5, 0.5, 0.2, 0.8]
    ];

    const scores = kVectors.map(k => qVector.reduce((acc, val, i) => acc + val * k[i], 0) / Math.sqrt(4));
    const expScores = scores.map(s => Math.exp(s));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const weights = expScores.map(s => s / sumExp);

    useEffect(() => {
        const timer = setInterval(() => {
            setStep(prev => (prev + 1) % 4);
        }, 3000);
        return () => clearInterval(timer);
    }, []);

    const renderVector = (vec: number[], label: string, color: string) => (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px' }}>
            <span style={{ fontSize: '10px', color: 'rgba(198, 216, 230, 0.6)' }}>{label}</span>
            <div style={{ display: 'flex', gap: '2px' }}>
                {vec.map((v, i) => (
                    <div key={i} style={{
                        width: '20px', height: '20px',
                        background: color,
                        opacity: 0.2 + Math.abs(v) * 0.8,
                        borderRadius: '2px',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '8px', color: '#fff'
                    }}>
                        {v.toFixed(1)}
                    </div>
                ))}
            </div>
        </div>
    );

    return (
        <div className="attention-microscope" style={{
            padding: '30px',
            background: 'rgba(42, 114, 163, 0.05)',
            backdropFilter: 'blur(15px)',
            borderRadius: '20px',
            border: '1px solid rgba(198, 216, 230, 0.1)',
            minHeight: '450px'
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
                <div>
                    <h3 style={{ color: '#c6d8e6', margin: 0 }}>🔬 Level 3: Attention Microscope</h3>
                    <p style={{ color: 'rgba(198, 216, 230, 0.6)', fontSize: '12px' }}>Nhìn sâu vào cơ chế tính toán sự chú ý của mô hình</p>
                </div>
                <div style={{ padding: '8px 15px', background: 'rgba(249, 114, 121, 0.1)', borderRadius: '20px', border: '1px solid #f9727933' }}>
                    <code style={{ color: '#f97279', fontSize: '13px' }}>{formula?.asciiMath}</code>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1px 1fr', gap: '40px', alignItems: 'center' }}>
                {/* Left Side: Operands */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                    <div style={{ opacity: step >= 0 ? 1 : 0.3, transition: 'all 0.5s' }}>
                        {renderVector(qVector, 'Query (Tôi đang tìm gì?)', '#f97279')}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', opacity: step >= 0 ? 1 : 0.3, transition: 'all 0.5s' }}>
                        <span style={{ fontSize: '10px', color: 'rgba(198, 216, 230, 0.6)', textAlign: 'center' }}>Keys (Thông tin có sẵn)</span>
                        {kVectors.map((k, i) => (
                            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                                {renderVector(k, `Key ${i}`, '#2a72a3')}
                                {step >= 1 && (
                                    <div style={{
                                        color: '#00ff88', fontSize: '12px', fontWeight: 'bold',
                                        animation: 'pulse 1s infinite'
                                    }}>
                                        Score: {scores[i].toFixed(2)}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Divider */}
                <div style={{ height: '80%', background: 'rgba(198, 216, 230, 0.1)' }}></div>

                {/* Right Side: Process & Weights */}
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '40px' }}>
                    {step >= 2 && (
                        <div style={{ animation: 'slideIn 0.5s ease-out' }}>
                            <h4 style={{ color: '#c6d8e6', fontSize: '14px', marginBottom: '15px' }}>Softmax Normalization</h4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                {weights.map((w, i) => (
                                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                        <div style={{ width: '60px', fontSize: '10px', color: 'rgba(198, 216, 230, 0.6)' }}>Token {i}</div>
                                        <div style={{ flex: 1, height: '12px', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', overflow: 'hidden' }}>
                                            <div style={{
                                                width: `${w * 100}%`, height: '100%',
                                                background: `linear-gradient(90deg, #2a72a3, #f97279)`,
                                                transition: 'width 1s cubic-bezier(0.34, 1.56, 0.64, 1)'
                                            }}></div>
                                        </div>
                                        <div style={{ width: '40px', fontSize: '11px', color: '#f97279', fontWeight: 'bold' }}>
                                            {(w * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {step >= 3 && (
                        <div style={{
                            padding: '20px', background: 'rgba(255,255,255,0.02)',
                            borderRadius: '12px', border: '1px solid rgba(198, 216, 230, 0.1)',
                            animation: 'fadeIn 1s'
                        }}>
                            <h4 style={{ color: '#c6d8e6', fontSize: '13px', marginBottom: '10px' }}>Ý nghĩa:</h4>
                            <p style={{ color: 'rgba(198, 216, 230, 0.7)', fontSize: '12px', lineHeight: '1.6' }}>
                                Mô hình quyết định tập trung vào <strong>Token {weights.indexOf(Math.max(...weights))}</strong> vì nó có sự tương đồng (Dot-product) cao nhất với Query hiện tại.
                                Các vector Value tương ứng sẽ được nhân với trọng số này để tạo ra kết quả cuối cùng.
                            </p>
                        </div>
                    )}
                </div>
            </div>

            <style jsx>{`
                @keyframes pulse {
                    0% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(1.1); opacity: 0.7; }
                    100% { transform: scale(1); opacity: 1; }
                }
                @keyframes slideIn {
                    from { transform: translateX(20px); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
            `}</style>
        </div>
    );
};
