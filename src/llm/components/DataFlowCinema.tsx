import React, { useState, useEffect, useRef } from 'react';

interface CinemaStep {
    id: number;
    title: string;
    description: string;
    icon: string;
}

const CINEMA_STEPS: CinemaStep[] = [
    { id: 0, title: 'Input Tokens', description: 'Dữ liệu thô bắt đầu dưới dạng các mảnh từ...', icon: '🔡' },
    { id: 1, title: 'Embedding Layer', description: 'Mỗi token biến thành một vector 768 chiều trong không gian ý nghĩa.', icon: '💎' },
    { id: 2, title: 'Attention Block', description: 'Các token "nói chuyện" với nhau để hiểu ngữ cảnh.', icon: '🧠' },
    { id: 3, title: 'FFN Block', description: 'Thông tin được xử lý sâu hơn qua các mạng thần kinh mật độ cao.', icon: '⚡' },
    { id: 4, title: 'Final Logits', description: 'Xác suất của từ tiếp theo hiện ra...', icon: '🔮' },
];

export const DataFlowCinema: React.FC = () => {
    const [activeStep, setActiveStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(true);
    const progressRef = useRef(0);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isPlaying) {
            interval = setInterval(() => {
                setActiveStep(prev => (prev + 1) % CINEMA_STEPS.length);
            }, 4000);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);

    return (
        <div className="data-flow-cinema" style={{
            padding: '40px',
            background: '#0a1a2a',
            borderRadius: '24px',
            border: '1px solid rgba(42, 114, 163, 0.3)',
            color: '#c6d8e6',
            minHeight: '600px',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
        }}>
            {/* Background Cinematic Grid */}
            <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                background: 'radial-gradient(circle at center, rgba(42, 114, 163, 0.1) 0%, transparent 70%)',
                zIndex: 0
            }} />

            {/* Cinematic Header */}
            <div style={{ position: 'relative', zIndex: 1, textAlign: 'center', marginBottom: '40px' }}>
                <span style={{ color: '#f97279', textTransform: 'uppercase', fontSize: '10px', letterSpacing: '3px', fontWeight: 'bold' }}>
                    LEVEL 4: CINEMA MODE
                </span>
                <h2 style={{ fontSize: '28px', marginTop: '10px', fontFamily: 'Inter, sans-serif' }}>
                    Hành Trình Của Một Ý Nghĩ
                </h2>
            </div>

            {/* Main Cinema View */}
            <div style={{
                flex: 1, position: 'relative', zIndex: 1,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                perspective: '1500px'
            }}>
                <div style={{
                    width: '100%', maxWidth: '800px', height: '300px',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-around',
                    transformStyle: 'preserve-3d'
                }}>
                    {CINEMA_STEPS.map((step, i) => {
                        const distance = Math.abs(i - activeStep);
                        const isPast = i < activeStep;
                        const isFuture = i > activeStep;

                        return (
                            <div key={i} style={{
                                width: '120px', height: '180px',
                                display: 'flex', flexDirection: 'column', alignItems: 'center',
                                transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
                                transform: `
                                    scale(${activeStep === i ? 1.2 : 0.8})
                                    translateZ(${activeStep === i ? 100 : -100}px)
                                    rotateY(${activeStep === i ? 0 : (isPast ? 45 : -45)}deg)
                                `,
                                opacity: activeStep === i ? 1 : (distance === 1 ? 0.4 : 0.1),
                                filter: activeStep === i ? 'drop-shadow(0 0 20px rgba(249, 114, 121, 0.4))' : 'none'
                            }}>
                                <div style={{
                                    fontSize: '50px', marginBottom: '20px',
                                    background: 'rgba(255,255,255,0.05)', padding: '20px',
                                    borderRadius: '50%', border: `2px solid ${activeStep === i ? '#f97279' : 'rgba(198, 216, 230, 0.2)'}`
                                }}>
                                    {step.icon}
                                </div>
                                <div style={{ fontWeight: 'bold', fontSize: '14px', textAlign: 'center' }}>{step.title}</div>
                            </div>
                        );
                    })}

                    {/* Glowing Data Line */}
                    <div style={{
                        position: 'absolute', top: '50%', left: '10%', right: '10%', height: '2px',
                        background: 'linear-gradient(90deg, transparent, #2a72a3, #f97279, transparent)',
                        zIndex: -1, opacity: 0.3
                    }} />
                </div>
            </div>

            {/* Subtitles & Controls */}
            <div style={{
                position: 'relative', zIndex: 1,
                padding: '30px', background: 'rgba(0,0,0,0.3)',
                borderRadius: '16px', border: '1px solid rgba(198, 216, 230, 0.1)',
                marginTop: '20px', backdropFilter: 'blur(20px)'
            }}>
                <div style={{ minHeight: '60px', textAlign: 'center' }}>
                    <h3 style={{ color: '#f97279', fontSize: '18px', marginBottom: '10px' }}>
                        {CINEMA_STEPS[activeStep].title}
                    </h3>
                    <p style={{ color: 'rgba(198, 216, 230, 0.8)', lineHeight: '1.6', fontSize: '14px' }}>
                        {CINEMA_STEPS[activeStep].description}
                    </p>
                </div>

                <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    gap: '20px', marginTop: '20px'
                }}>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        style={{
                            background: '#2a72a3', border: 'none', color: '#fff',
                            width: '40px', height: '40px', borderRadius: '50%',
                            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}
                    >
                        {isPlaying ? '⏸' : '▶'}
                    </button>

                    <div style={{ display: 'flex', gap: '8px' }}>
                        {CINEMA_STEPS.map((_, i) => (
                            <div
                                key={i}
                                onClick={() => { setActiveStep(i); setIsPlaying(false); }}
                                style={{
                                    width: '30px', height: '4px',
                                    background: activeStep === i ? '#f97279' : 'rgba(198, 216, 230, 0.2)',
                                    borderRadius: '2px', cursor: 'pointer', transition: 'all 0.3s'
                                }}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <style jsx>{`
                .data-flow-cinema::before {
                    content: "";
                    position: absolute;
                    top: 0; left: 0; right: 0; height: 1px;
                    background: linear-gradient(90deg, transparent, rgba(198, 216, 230, 0.3), transparent);
                }
            `}</style>
        </div>
    );
};
