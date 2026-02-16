import React, { useState } from 'react';

interface BenchmarkScore {
    name: string;
    score: number;
    color: string;
}

export const EvaluationCenter: React.FC = () => {
    const [benchmarks] = useState<BenchmarkScore[]>([
        { name: 'MMLU (Kiến thức tổng hợp)', score: 86.4, color: '#f97279' },
        { name: 'HumanEval (Viết Code)', score: 67.0, color: '#2a72a3' },
        { name: 'GSM8K (Toán học)', score: 92.0, color: '#c6d8e6' },
        { name: 'TruthfulQA (Độ trung thực)', score: 59.4, color: '#f97279' },
    ]);

    const [isDeploying, setIsDeploying] = useState(false);
    const [deployStatus, setDeployStatus] = useState(0);

    const handleDeploy = () => {
        setIsDeploying(true);
        const interval = setInterval(() => {
            setDeployStatus(prev => {
                if (prev >= 100) {
                    clearInterval(interval);
                    return 100;
                }
                return prev + 5;
            });
        }, 100);
    };

    return (
        <div className="evaluation-center" style={{ padding: '30px', background: 'rgba(255, 255, 255, 0.03)', backdropFilter: 'blur(10px)', borderRadius: '16px', border: '1px solid rgba(198, 216, 230, 0.1)' }}>
            <h3 style={{ color: '#c6d8e6', marginBottom: '25px', fontFamily: 'Inter, sans-serif' }}>✅ Stage 5: Evaluation & Deployment</h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
                {/* Benchmarks Section */}
                <div>
                    <h4 style={{ color: '#f97279', fontSize: '14px', marginBottom: '15px' }}>📊 Kiểm Tra IQ (Benchmarks)</h4>
                    <div style={{ display: 'grid', gap: '15px' }}>
                        {benchmarks.map((b, i) => (
                            <div key={i}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '12px', color: '#c6d8e6' }}>
                                    <span>{b.name}</span>
                                    <span style={{ fontWeight: 'bold' }}>{b.score}%</span>
                                </div>
                                <div style={{ width: '100%', height: '8px', background: 'rgba(0,0,0,0.3)', borderRadius: '4px' }}>
                                    <div style={{
                                        width: `${b.score}%`,
                                        height: '100%',
                                        background: b.color,
                                        borderRadius: '4px',
                                        boxShadow: `0 0 10px ${b.color}44`
                                    }}></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Deployment Section */}
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '20px', background: 'rgba(42, 114, 163, 0.1)', borderRadius: '12px', border: '1px solid rgba(198, 216, 230, 0.05)' }}>
                    <div style={{ fontSize: '48px', marginBottom: '20px' }}>🚀</div>
                    <h4 style={{ color: '#c6d8e6', marginBottom: '10px' }}>Sẵn Sàng Triển Khai?</h4>
                    <p style={{ color: 'rgba(198, 216, 230, 0.6)', fontSize: '12px', textAlign: 'center', marginBottom: '20px' }}>
                        Mô hình của bạn đã vượt qua các bài kiểm tra an toàn và độ chính xác cao.
                    </p>

                    {!isDeploying ? (
                        <button
                            onClick={handleDeploy}
                            style={{
                                background: '#f97279', color: '#fff', border: 'none', padding: '12px 30px',
                                borderRadius: '30px', fontWeight: 'bold', cursor: 'pointer',
                                boxShadow: '0 4px 15px rgba(249, 114, 121, 0.3)',
                                transition: 'all 0.3s'
                            }}
                            onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                            onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
                        >
                            DEPLOY TO CLOUD
                        </button>
                    ) : (
                        <div style={{ width: '100%', textAlign: 'center' }}>
                            <div style={{ color: '#c6d8e6', fontSize: '14px', marginBottom: '10px' }}>Đang nén mô hình & tải lên: {deployStatus}%</div>
                            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                                <div style={{ width: `${deployStatus}%`, height: '100%', background: '#00ff88', borderRadius: '2px', transition: 'width 0.1s' }}></div>
                            </div>
                            {deployStatus === 100 && (
                                <div style={{ color: '#00ff88', marginTop: '10px', fontSize: '12px', animation: 'fadeIn 0.5s' }}>
                                    ✅ Hoàn thành! Model hiện đã có mặt tại: <strong>api.yourmodel.ai</strong>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
            `}</style>
        </div>
    );
};
