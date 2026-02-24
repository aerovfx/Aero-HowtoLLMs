'use client';
import { faBook, faChartLine, faCode, faCog, faGraduationCap, faRobot } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import React from 'react';
import s from './HomePage.module.scss';

interface CourseModule {
    id: string;
    title: string;
    description: string;
    icon: any;
    lessons: string[];
}

const courseModules: CourseModule[] = [
    {
        id: 'foundations',
        title: 'Phần 1: Nền Tảng Kiến Trúc',
        description: 'Từ Transformer đến các mô hình LLM hiện đại',
        icon: faCog,
        lessons: [
            'Bài 00: Tổng quan về LLM',
            'Bài 01: Transformer Architecture',
            'Bài 02: Transformer Tricks & Optimizations',
            'Bài 03: Giải mã các mô hình LLM'
        ]
    },
    {
        id: 'building',
        title: 'Phần 2: Xây Dựng & Tinh Chỉnh',
        description: 'Huấn luyện và tối ưu hóa mô hình',
        icon: faCode,
        lessons: [
            'Bài 04: Training & Pre-training',
            'Bài 05: Fine-tuning & PEFT (LoRA, QLoRA)'
        ]
    },
    {
        id: 'advanced',
        title: 'Phần 3: Khả Năng Nâng Cao',
        description: 'Biến LLM thành Agent thông minh',
        icon: faRobot,
        lessons: [
            'Bài 06: Reasoning & Prompt Engineering',
            'Bài 07: Agentic LLMs & Tool Use'
        ]
    },
    {
        id: 'evaluation',
        title: 'Phần 4: Đánh Giá & Công Cụ',
        description: 'Đo lường và cải thiện hiệu suất',
        icon: faChartLine,
        lessons: [
            'Bài 08: Evaluation & Benchmarks',
            'Bài 09: Recap & Trends',
            'Bài 10: Essential Tools for AI Engineers'
        ]
    }
];

export const HomePage: React.FC = () => {
    let router = useRouter();

    return <div className={s.homePage}>
        {/* Hero Section */}
        <div className={s.heroSection}>
            <div className={s.heroContent}>
                <div className={s.heroIcon}>
                    <FontAwesomeIcon icon={faGraduationCap} />
                </div>
                <h1 className={s.heroTitle}>
                    Khóa học: Transformers & Large Language Models
                </h1>
                <p className={s.heroSubtitle}>
                    Hướng dẫn chuyên sâu về LLM - Từ nền tảng Transformer đến xây dựng AI Agent hiện đại
                </p>
                <p className={s.heroDescription}>
                    Dựa trên giáo trình Stanford CME 295, biên soạn bởi Pixiboss
                </p>
            </div>
        </div>

        {/* Interactive Visualization */}
        <div className={s.visualizationSection}>
            <div className={s.projectCard} onClick={() => router.push('/llm')}>
                <div className={s.cardImageWrapper}>
                    <div className={s.cardImage}>
                        <img src="/images/llm-viz-screenshot2.png" alt="LLM Visualization Screenshot" />
                    </div>
                </div>
                <div className={s.cardContent}>
                    <div className={s.cardTitle}>
                        <Link href={"/llm"}>
                            <FontAwesomeIcon icon={faBook} className={s.cardIcon} />
                            Trực quan hóa LLM 3D
                        </Link>
                    </div>
                    <div className={s.cardText}>
                        Khám phá thuật toán LLM (GPT) trong môi trường 3D tương tác.
                        Xem từng phép cộng, nhân và quá trình hoạt động thực tế của mô hình.
                    </div>
                </div>
            </div>
        </div>

        {/* Learning Path */}
        <div className={s.learningPath}>
            <h2 className={s.sectionTitle}>Lộ Trình Học Tập</h2>
            <div className={s.pathDescription}>
                Khóa học được thiết kế theo 4 giai đoạn, giúp bạn nắm vững từ cơ bản đến nâng cao
            </div>
            <div className={s.modulesGrid}>
                {courseModules.map((module, idx) => (
                    <div key={module.id} className={s.moduleCard}>
                        <div className={s.moduleHeader}>
                            <div className={s.moduleIcon}>
                                <FontAwesomeIcon icon={module.icon} />
                            </div>
                            <div className={s.moduleNumber}>0{idx + 1}</div>
                        </div>
                        <h3 className={s.moduleTitle}>{module.title}</h3>
                        <p className={s.moduleDescription}>{module.description}</p>
                        <ul className={s.lessonList}>
                            {module.lessons.map((lesson, i) => (
                                <li key={i}>{lesson}</li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
        </div>

        {/* Key Features */}
        <div className={s.featuresSection}>
            <h2 className={s.sectionTitle}>Nội Dung Chính</h2>
            <div className={s.featuresGrid}>
                <div className={s.featureCard}>
                    <h3>Đào Tạo Từ Đầu</h3>
                    <p>Học cách pre-training mô hình từ con số 0, Scaling Laws và quản lý dữ liệu lớn</p>
                </div>
                <div className={s.featureCard}>
                    <h3>Fine-tuning Hiệu Quả</h3>
                    <p>Nắm vững LoRA, QLoRA, Prompt Tuning và RLHF để tối ưu chi phí</p>
                </div>
                <div className={s.featureCard}>
                    <h3>Agentic AI</h3>
                    <p>Biến LLM thành Agent tự chủ biết sử dụng công cụ và RAG</p>
                </div>
                <div className={s.featureCard}>
                    <h3>Công Cụ Thực Tế</h3>
                    <p>Top 12 repo quan trọng: vLLM, llama.cpp, Unsloth và nhiều hơn nữa</p>
                </div>
            </div>
        </div>

        {/* Footer */}
        <div className={s.footerSection}>
            <p className={s.footerText}>
                Tài liệu được lưu trữ tại <code>docs/01-01-LLM_Course</code> của repository Aero-HowtoLLMs
            </p>
        </div>
    </div>;
}
