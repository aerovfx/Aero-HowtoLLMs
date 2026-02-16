import React from 'react';
import { LayerView } from '@/src/llm/LayerView';
import { InfoButton } from '@/src/llm/WelcomePopup';
import { Header } from '@/src/homepage/Header';

export const metadata = {
    title: 'LLM Visualization',
    description: 'A 3D animated visualization of an LLM with a walkthrough.',
};

export default function Page() {
    return <>
        <Header title="LLM Visualization">
            <InfoButton />
        </Header>
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
