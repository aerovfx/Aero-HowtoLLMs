/**
 * Vietnamese Font Atlas Generator (TypeScript)
 * Công cụ tạo Font Atlas hỗ trợ đầy đủ ký tự tiếng Việt cho WebGL.
 */

import fs from 'fs';
import path from 'path';
// @ts-ignore
import generateBMFont from 'msdf-bmfont-xml';

const VIETNAMESE_CHARSET =
    '! "#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~' +
    'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ' +
    'ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ' +
    '‧\\—Σγβσμε';

const CONFIG = {
    fontPath: 'fonts/Roboto-Regular.ttf',
    outputDir: 'public/fonts',
    textureSize: [1024, 1024],
    fontSize: 42,
    fieldType: 'msdf'
};

async function build() {
    console.log('🚀 Bắt đầu tạo Vietnamese Font Atlas...');

    return new Promise((resolve, reject) => {
        generateBMFont(CONFIG.fontPath, {
            charset: VIETNAMESE_CHARSET,
            fieldType: CONFIG.fieldType,
            textureSize: CONFIG.textureSize,
            fontSize: CONFIG.fontSize,
            outputType: 'json'
        }, (error: any, textures: any, font: any) => {
            if (error) {
                console.error('❌ Lỗi:', error);
                reject(error);
                return;
            }

            if (!fs.existsSync(CONFIG.outputDir)) {
                fs.mkdirSync(CONFIG.outputDir, { recursive: true });
            }

            // Lưu texture
            textures.forEach((tex: any, i: number) => {
                const texPath = path.join(CONFIG.outputDir, `font-atlas-${i}.png`);
                fs.writeFileSync(texPath, tex.texture);
                console.log(`✅ Đã lưu texture: ${texPath}`);
            });

            // Lưu font definition
            const fontPath = path.join(CONFIG.outputDir, 'font-def-vietnamese.json');
            fs.writeFileSync(fontPath, font.data);
            console.log(`✅ Đã lưu cấu trúc font: ${fontPath}`);

            resolve(true);
        });
    });
}

// Chạy nếu trực tiếp
if (require.main === module) {
    build().catch(console.error);
}

export { build };
