import React, { useState, useEffect, useRef } from 'react';

export const TrainingGymnasium: React.FC = () => {
    const [step, setStep] = useState(0);
    const [lossData, setLossData] = useState<number[]>([]);
    const [pplData, setPplData] = useState<number[]>([]);
    const [gpuActivity, setGpuActivity] = useState<number[]>([85, 72, 90, 88, 76, 92, 84, 89]);
    const maxSteps = 1000;

    // Giả lập dữ liệu huấn luyện
    useEffect(() => {
        const interval = setInterval(() => {
            setStep(prev => {
                if (prev >= maxSteps) return prev;
                const nextStep = prev + 1;

                // Công thức giả lập Loss giảm dần theo đường cong logarit + nhiễu
                const currentLoss = 8 * Math.exp(-nextStep / 300) + Math.random() * 0.2;
                const currentPpl = 50 * Math.exp(-nextStep / 400) + 1.2 + Math.random() * 0.5;

                setLossData(prevData => [...prevData.slice(-40), currentLoss]);
                setPplData(prevData => [...prevData.slice(-40), currentPpl]);

                // Giả lập GPU thay đổi tải
                setGpuActivity(prev => prev.map(val => Math.min(100, Math.max(70, val + (Math.random() - 0.5) * 10))));

                return nextStep;
            });
        }, 150);

        return () => clearInterval(interval);
    }, []);

    const renderChart = (data: number[], color: string, label: string, currentVal: string) => {
        const height = 100;
        const width = 300;
        const points = data.map((d, i) => `${(i * (width / 40))},${height - (d * (height / Math.max(...data, 10)))}`).join(' ');

        return (
            <div className="chart-box" style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '15px', borderRadius: '12px', border: '1px solid rgba(198, 216, 230, 0.1)', flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <span style={{ color: '#c6d8e6', fontSize: '12px', fontWeight: 'bold' }}>{label}</span>
                    <span style={{ color: color, fontSize: '14px', fontWeight: '800' }}>{currentVal}</span>
                </div>
                <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                    <polyline fill="none" stroke={color} strokeWidth="2" points={points} style={{ transition: 'all 0.2s' }} />
                    <path d={`M ${points} L ${width} ${height} L 0 ${height} Z`} fill={`url(#grad-${label})`} opacity="0.2" />
                    <defs>
                        <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={color} />
                            <stop offset="100%" stopColor="transparent" />
                        </linearGradient>
                    </defs>
                </svg>
            </div>
        );
    };

    return (
        <div className="training-gym" style={{ padding: '30px', background: 'rgba(42, 114, 163, 0.05)', backdropFilter: 'blur(10px)', borderRadius: '16px', border: '1px solid rgba(198, 216, 230, 0.1)' }}>
            <h3 style={{ color: '#c6d8e6', marginBottom: '20px', fontFamily: 'Inter, sans-serif' }}>🎓 Training Gymnasium: Pre-training Phase</h3>

            {/* Progress Bar */}
            <div style={{ marginBottom: '30px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'rgba(198, 216, 230, 0.6)', fontSize: '11px', marginBottom: '8px' }}>
                    <span>Training Progress: Step {step} / {maxSteps}</span>
                    <span>{((step / maxSteps) * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '6px', background: 'rgba(0,0,0,0.3)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ width: `${(step / maxSteps) * 100}%`, height: '100%', background: 'linear-gradient(to right, #2a72a3, #f97279)', transition: 'width 0.3s' }}></div>
                </div>
            </div>

            {/* Charts Row */}
            <div style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
                {renderChart(lossData, '#f97279', 'Loss', lossData[lossData.length - 1]?.toFixed(4) || '---')}
                {renderChart(pplData, '#2a72a3', 'Perplexity', pplData[pplData.length - 1]?.toFixed(2) || '---')}
            </div>

            {/* GPU Cluster Activity */}
            <div>
                <div style={{ color: '#c6d8e6', fontSize: '12px', fontWeight: 'bold', marginBottom: '15px' }}>⚡ GPU Cluster Activity (H100)</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '10px' }}>
                    {gpuActivity.map((val, i) => (
                        <div key={i} style={{ textAlign: 'center' }}>
                            <div style={{ height: '60px', width: '100%', background: 'rgba(0,0,0,0.2)', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                                <div style={{
                                    position: 'absolute', bottom: 0, width: '100%',
                                    height: `${val}%`,
                                    background: val > 85 ? '#f97279' : '#2a72a3',
                                    transition: 'all 0.5s',
                                    opacity: 0.8
                                }}></div>
                            </div>
                            <span style={{ fontSize: '9px', color: 'rgba(198, 216, 230, 0.4)', marginTop: '4px', display: 'block' }}>GPU-{i}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div style={{ marginTop: '20px', padding: '15px', background: 'rgba(249, 114, 121, 0.05)', borderRadius: '8px', borderLeft: '3px solid #f97279' }}>
                <p style={{ fontSize: '13px', color: '#c6d8e6', margin: 0 }}>
                    💡 <strong>Mẹo:</strong> Perplexity đo lường độ bối rối của mô hình. Giá trị càng giảm, mô hình càng trở nên "thông minh" và chắc chắn hơn về dự đoán của mình.
                </p>
            </div>
        </div>
    );
};
