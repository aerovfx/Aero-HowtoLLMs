import React from 'react';
import { Phase } from "./Walkthrough";
import { commentary, embed, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Vec3 } from '@/src/utils/vector';

let minGptLink = 'https://github.com/karpathy/minGPT';
let pytorchLink = 'https://pytorch.org/';
let andrejLink = 'https://karpathy.ai/';
let zeroToHeroLink = 'https://karpathy.ai/zero-to-hero.html';

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    setInitialCamera(state, new Vec3(184.744, 0.000, -636.820), new Vec3(296.000, 16.000, 13.500));

    let c0 = commentary(wt, null, 0)`
Tr\u01b0\u1edbc khi \u0111i s\u00e2u v\u00e0o chi ti\u1ebft thu\u1eadt to\u00e1n, h\u00e3y l\u00f9i l\u1ea1i m\u1ed9t ch\u00fat.

H\u01b0\u1edbng d\u1eabn n\u00e0y t\u1eadp trung v\u00e0o _suy lu\u1eadn_ (inference), kh\u00f4ng ph\u1ea3i hu\u1ea5n luy\u1ec7n, v\u00e0 do \u0111\u00f3 ch\u1ec9 l\u00e0 m\u1ed9t ph\u1ea7n nh\u1ecf trong to\u00e0n b\u1ed9 qu\u00e1 tr\u00ecnh h\u1ecdc m\u00e1y.
Trong tr\u01b0\u1eddng h\u1ee3p c\u1ee7a ch\u00fang ta, c\u00e1c tr\u1ecdng s\u1ed1 c\u1ee7a m\u00f4 h\u00ecnh \u0111\u00e3 \u0111\u01b0\u1ee3c hu\u1ea5n luy\u1ec7n tr\u01b0\u1edbc, v\u00e0 ch\u00fang ta s\u1eed d\u1ee5ng qu\u00e1 tr\u00ecnh suy lu\u1eadn \u0111\u1ec3 t\u1ea1o \u0111\u1ea7u ra. Qu\u00e1 tr\u00ecnh n\u00e0y ch\u1ea1y tr\u1ef1c ti\u1ebfp trong tr\u00ecnh duy\u1ec7t c\u1ee7a b\u1ea1n.

M\u00f4 h\u00ecnh \u0111\u01b0\u1ee3c tr\u00ecnh b\u00e0y \u1edf \u0111\u00e2y l\u00e0 m\u1ed9t ph\u1ea7n c\u1ee7a h\u1ecd GPT (generative pre-trained transformer), c\u00f3 th\u1ec3 \u0111\u01b0\u1ee3c m\u00f4 t\u1ea3 l\u00e0 \"b\u1ed9 d\u1ef1 \u0111o\u00e1n token d\u1ef1a tr\u00ean ng\u1eef c\u1ea3nh\".
OpenAI \u0111\u00e3 gi\u1edbi thi\u1ec7u h\u1ecd n\u00e0y v\u00e0o n\u0103m 2018, v\u1edbi c\u00e1c th\u00e0nh vi\u00ean n\u1ed5i b\u1eadt nh\u01b0 GPT-2, GPT-3 v\u00e0 GPT-3.5 Turbo, sau c\u00f9ng l\u00e0 n\u1ec1n t\u1ea3ng c\u1ee7a ChatGPT \u0111\u01b0\u1ee3c s\u1eed d\u1ee5ng r\u1ed9ng r\u00e3i.
N\u00f3 c\u0169ng c\u00f3 th\u1ec3 li\u00ean quan \u0111\u1ebfn GPT-4, nh\u01b0ng chi ti\u1ebft c\u1ee5 th\u1ec3 v\u1eabn ch\u01b0a \u0111\u01b0\u1ee3c bi\u1ebft.

H\u01b0\u1edbng d\u1eabn n\u00e0y l\u1ea5y c\u1ea3m h\u1ee9ng t\u1eeb d\u1ef1 \u00e1n GitHub ${embedLink('minGPT', minGptLink)}, m\u1ed9t tri\u1ec3n khai GPT t\u1ed1i gi\u1ea3n trong ${embedLink('PyTorch', pytorchLink)}
do ${embedLink('Andrej Karpathy', andrejLink)} t\u1ea1o ra.
S\u00ea-ri YouTube c\u1ee7a anh \u1ea5y ${embedLink("Neural Networks: Zero to Hero", zeroToHeroLink)} v\u00e0 d\u1ef1 \u00e1n minGPT \u0111\u00e3 l\u00e0 ngu\u1ed3n t\u00e0i nguy\u00ean v\u00f4 gi\u00e1 trong vi\u1ec7c t\u1ea1o ra
h\u01b0\u1edbng d\u1eabn n\u00e0y. M\u00f4 h\u00ecnh m\u1eabu trong \u0111\u00e2y d\u1ef1a tr\u00ean m\u1ed9t m\u00f4 h\u00ecnh t\u00ecm th\u1ea5y trong d\u1ef1 \u00e1n minGPT.

Th\u00f4i, h\u00e3y b\u1eaft \u0111\u1ea7u!
`;

}

export function embedLink(a: React.ReactNode, href: string) {
    return embedInline(<a className={s.externalLink} href={href} target="_blank" rel="noopener noreferrer">{a}</a>);
}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}


// Another similar model is BERT (bidirectional encoder representations from transformers), a "context-aware text encoder" commonly
// used for tasks like document classification and search.  Newer models like Facebook's LLaMA (large language model architecture), continue to use
// a similar transformer architecture, albeit with some minor differences.
