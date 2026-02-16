import React, { useState } from 'react';
import { PipelineStage } from '../VisualEngine';
import { PipelineJourneyMap } from './PipelineJourneyMap';
import { TokenizationLab } from './TokenizationLab';
import { ArchitectureConstruction } from './ArchitectureConstruction';
import { TrainingGymnasium } from './TrainingGymnasium';
import { EvaluationCenter } from './EvaluationCenter';
import { AttentionMicroscope } from './AttentionMicroscope';
import { DataFlowCinema } from './DataFlowCinema';
import { LLMPlayground } from './LLMPlayground';

export const LLMCreationPipeline: React.FC = () => {
    const [activeStage, setActiveStage] = useState<PipelineStage>(PipelineStage.DataCollection);
    const [viewMode, setViewMode] = useState<'normal' | 'microscope' | 'cinema' | 'playground'>('normal');

    const renderStageContent = () => {
        // Handle Level 5: Playground Mode
        if (viewMode === 'playground') {
            return (
                <div style={{ position: 'relative', animation: 'fadeIn 1s' }}>
                    <button
                        onClick={() => setViewMode('normal')}
                        style={{
                            position: 'absolute', top: '15px', right: '15px',
                            background: 'rgba(249, 114, 121, 0.2)', color: '#f97279',
                            border: '1px solid #f97279', padding: '6px 15px',
                            borderRadius: '20px', fontSize: '11px', cursor: 'pointer',
                            backdropFilter: 'blur(10px)', zIndex: 10
                        }}
                    >
                        ✕ Đóng Playground
                    </button>
                    <LLMPlayground />
                </div>
            );
        }

        // Handle Level 4: Cinema Mode
        if (viewMode === 'cinema') {
            return (
                <div style={{ position: 'relative', animation: 'fadeIn 1s' }}>
                    <button
                        onClick={() => setViewMode('normal')}
                        style={{
                            position: 'absolute', top: '15px', right: '15px',
                            background: 'rgba(249, 114, 121, 0.2)', color: '#f97279',
                            border: '1px solid #f97279', padding: '6px 15px',
                            borderRadius: '20px', fontSize: '11px', cursor: 'pointer',
                            backdropFilter: 'blur(10px)', zIndex: 10
                        }}
                    >
                        ✕ Đóng Cinema Mode
                    </button>
                    <DataFlowCinema />
                </div>
            );
        }

        // Handle Level 3: Microscope zoom
        if (viewMode === 'microscope' && activeStage === PipelineStage.ArchitectureDesign) {
            return (
                <div style={{ position: 'relative', animation: 'zoomIn 0.5s' }}>
                    <button
                        onClick={() => setViewMode('normal')}
                        style={{
                            position: 'absolute', top: '10px', right: '10px',
                            background: 'rgba(42, 114, 163, 0.2)', color: '#c6d8e6',
                            border: '1px solid rgba(198, 216, 230, 0.3)', padding: '6px 15px',
                            borderRadius: '20px', fontSize: '11px', cursor: 'pointer',
                            backdropFilter: 'blur(10px)', zIndex: 10
                        }}
                    >
                        ← Trở lại Kiến trúc
                    </button>
                    <AttentionMicroscope />
                </div>
            );
        }

        switch (activeStage) {
            case PipelineStage.Evaluation:
                return <EvaluationCenter />;
            case PipelineStage.PreTraining:
                return <TrainingGymnasium />;
            case PipelineStage.ArchitectureDesign:
                return (
                    <div style={{ position: 'relative' }}>
                        <ArchitectureConstruction />
                        <div style={{ position: 'absolute', bottom: '20px', left: '20px', right: '20px', display: 'flex', justifyContent: 'center' }}>
                            <button
                                onClick={() => setViewMode('microscope')}
                                style={{
                                    background: 'rgba(249, 114, 121, 0.2)', color: '#f97279',
                                    border: '1px solid #f97279', padding: '10px 25px',
                                    borderRadius: '30px', fontSize: '13px', fontWeight: 'bold',
                                    cursor: 'pointer', backdropFilter: 'blur(10px)',
                                    boxShadow: '0 0 20px rgba(249, 114, 121, 0.2)',
                                    transition: 'all 0.3s'
                                }}
                                onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            >
                                🔬 Zoom vào Cơ chế Attention (Level 3)
                            </button>
                        </div>
                    </div>
                );
            case PipelineStage.PreProcessing:
                return <TokenizationLab />;
            case PipelineStage.DataCollection:
                return (
                    <div style={{ padding: '40px', color: '#c6d8e6', textAlign: 'center', background: 'rgba(255, 255, 255, 0.02)', border: '1px dashed rgba(198, 216, 230, 0.2)', borderRadius: '16px' }}>
                        <h2 style={{ color: '#f97279', fontFamily: 'Inter, sans-serif', fontWeight: '700' }}>🕷️ Giai đoạn: Thu Thập Dữ Liệu</h2>
                        <p style={{ opacity: 0.8, maxWidth: '600px', margin: '20px auto', lineHeight: '1.6' }}>Hệ thống đang mô phỏng việc thu thập 13 Trillion tokens từ Common Crawl, Wikipedia và GitHub...</p>
                        <div style={{ marginTop: '30px', fontSize: '48px', filter: 'drop-shadow(0 0 10px rgba(42, 114, 163, 0.5))' }}>🌐 ➔ 📄 ➔ 💾</div>
                    </div>
                );
            default:
                return (
                    <div style={{ padding: '60px', textAlign: 'center', color: 'rgba(198, 216, 230, 0.5)', background: 'rgba(0, 0, 0, 0.1)', borderRadius: '16px' }}>
                        <p>Nội dung trực quan hóa cho stage <strong style={{ color: '#f97279' }}>{PipelineStage[activeStage]}</strong> đang được phát triển...</p>
                    </div>
                );
        }
    };

    return (
        <div className="llm-creation-pipeline" style={{ padding: '40px', color: '#c6d8e6', maxWidth: '1200px', margin: '0 auto' }}>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '20px', marginBottom: '50px' }}>
                <h1 style={{ fontFamily: 'Inter, sans-serif', fontWeight: '800', margin: 0, background: 'linear-gradient(135deg, #c6d8e6 0%, #f97279 50%, #2a72a3 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontSize: '42px', letterSpacing: '-1px' }}>
                    QUY TRÌNH TẠO MÔ HÌNH LLM
                </h1>
                <button
                    onClick={() => setViewMode(viewMode === 'cinema' ? 'normal' : 'cinema')}
                    style={{
                        background: viewMode === 'cinema' ? 'rgba(249, 114, 121, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                        color: '#c6d8e6',
                        border: viewMode === 'cinema' ? '1px solid #f97279' : '1px solid rgba(198, 216, 230, 0.2)',
                        padding: '10px 20px',
                        borderRadius: '30px', fontSize: '13px', cursor: 'pointer',
                        backdropFilter: 'blur(10px)', transition: 'all 0.3s',
                        display: 'flex', alignItems: 'center', gap: '8px'
                    }}
                >
                    🎬 Cinema Mode
                </button>
                <button
                    onClick={() => setViewMode(viewMode === 'playground' ? 'normal' : 'playground')}
                    style={{
                        background: viewMode === 'playground' ? 'rgba(42, 114, 163, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                        color: '#c6d8e6',
                        border: viewMode === 'playground' ? '1px solid #2a72a3' : '1px solid rgba(198, 216, 230, 0.2)',
                        padding: '10px 20px',
                        borderRadius: '30px', fontSize: '13px', cursor: 'pointer',
                        backdropFilter: 'blur(10px)', transition: 'all 0.3s',
                        display: 'flex', alignItems: 'center', gap: '8px'
                    }}
                >
                    🎮 Playground
                </button>
            </div>

            {/* Level 1: Journey Map */}
            <div style={{ marginBottom: '50px' }}>
                <PipelineJourneyMap onStageSelect={setActiveStage} />
            </div>

            {/* Level 2: Stage Detail */}
            <div className="stage-content-container" style={{ minHeight: '450px', animation: 'fadeIn 0.6s cubic-bezier(0.23, 1, 0.32, 1)' }}>
                {renderStageContent()}
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
};
