let fs = require('fs');
let path = require('path');
let generateBMFont = require('msdf-bmfont-xml');

function generateBMFontP(name, opts) {
    return new Promise((resolve, reject) => {
        console.log('=== generating', name);
        generateBMFont(name, opts, (error, textures, font) => {
            if (error) {
                reject(error);
            } else {
                resolve([textures, font]);
            }
        });
    });
}

async function generateAndSave(name, opts, sharedName) {

    let [textures, font] = await generateBMFontP(name, opts);

    let idx = 0;
    for (let tex of textures) {
        console.log('saving texture', tex.filename + '.png');
        fs.writeFileSync(tex.filename + '.png', tex.texture);
        idx++;
    }

    console.log('saving font json', font.filename);
    fs.writeFileSync(font.filename, font.data);

    console.log('saving font cfg', opts.reuse);
    fs.writeFileSync(opts.reuse, JSON.stringify(font.settings, null, '\t'));

    return {
        name: path.basename(name, path.extname(name)),
        fontDefFile: path.normalize(font.filename),
        sharedName,
    };
}

let commonOpts = {
    fieldType: 'msdf',
    outputType: 'json',
    filename: 'fonts/font-atlas.png',
    reuse: 'fonts/font-atlas.cfg',
    textureSize: [1024, 1024],
};

async function runAll() {
    fs.rmSync(commonOpts.reuse, { force: true });

    let files = [];

    files.push(await generateAndSave('fonts/Roboto-Regular.ttf', {
        ...commonOpts,
        charset: '! "#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~' + 
                 '\u2022\u2013\u2014\u03a3\u03b3\u03b2\u03c3\u03bc\u03b5' + // symbols & greek
                 '\u00e1\u00e0\u1ea3\u00e3\u1ea1\u0103\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u00e2\u1ea5\u1ead\u1ea7\u1ea9\u1eab' + // a
                 '\u00e9\u00e8\u1ebb\u1ebd\u1eb9\u00ea\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7' + // e
                 '\u00ed\u00ec\u1ec9\u0129\u1ecb' + // i
                 '\u00f3\u00f2\u1ecf\u00f5\u1ecd\u00f4\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u01a1\u1edb\u1edd\u1edf\u1ee1\u1ee3' + // o
                 '\u00fa\u00f9\u1ee7\u0169\u1ee5\u01b0\u1ee9\u1eeb\u1eed\u1eef\u1ef1' + // u
                 '\u00fd\u1ef3\u1ef7\u1ef9\u1ef5\u0111' + // y & d
                 '\u00c1\u00c0\u1ea2\u00c3\u1ea0\u0102\u1eae\u1eb0\u1eb2\u1eb4\u1eb6\u00c2\u1ea4\u1ea6\u1ea8\u1eaa\u1eac' + // A
                 '\u00c9\u00c8\u1eba\u1ebc\u1eb8\u00ca\u1ebe\u1ec0\u1ec2\u1ec4\u1ec6' + // E
                 '\u00cd\u00cc\u1ec8\u0128\u1eca' + // I
                 '\u00d3\u00d2\u1ece\u00d5\u1ecc\u00d4\u1ed0\u1ed2\u1ed4\u1ed6\u1ed8\u01a0\u1eda\u1edc\u1ede\u1ee0\u1ee2' + // O
                 '\u00da\u00d9\u1ee6\u0168\u1ee4\u01af\u1ee8\u1eea\u1eec\u1eee\u1ef0' + // U
                 '\u00dd\u1ef2\u1ef6\u1ef8\u1ef4\u0110' + // Y & D
                 '\u00b7\u2212\u00d7\u00f7\u2299\u2211\u1d62\u1d63\u2098\u2099\u2080\u2081\u2082', // operators & subscripts
    }, 'regular'));

    // math italic
    files.push(await generateAndSave('fonts/cmmi12.ttf', {
        ...commonOpts,
        charset: 'xyztcbXYZTCB',
    }, 'math'));

    // symbols
    files.push(await generateAndSave('fonts/cmsy10.ttf', {
        ...commonOpts,
        charset: '-+/()\u00a3\u0070\u00a1\u006a',
    }, 'math'));

    // operators
    files.push(await generateAndSave('fonts/cmr12.ttf', {
        ...commonOpts,
        charset: '=?;:,-.',
    }, 'math'));

    combineAndCopyToOutput(files)
}

function combineAndCopyToOutput(files) {
    let fontData = {
        faces: [],
    };

    for (let { name, fontDefFile, sharedName } of files) {
        let fileContents = fs.readFileSync(fontDefFile, { encoding: 'utf8' });

        let fontSrc = JSON.parse(fileContents);

        fontData.pages = fontSrc.pages;

        let face = {
            name: name,
            common: fontSrc.common,
            info: { ...fontSrc.info,
                charset: fontSrc.info.charset.join(''),
            },
        };

        {
            let charArr = new Int16Array(fontSrc.chars.length * 12);

            let index = 0;
            for (let c of fontSrc.chars) {
                let order = [c.id,
                    c.index,
                    c.char.codePointAt(0),
                    c.x,
                    c.y,
                    c.width,
                    c.height,
                    c.xoffset,
                    c.yoffset,
                    c.xadvance,
                    c.page,
                    c.chnl];

                for (let x of order) {
                    charArr[index++] = x;
                }
            }

            face.chars = Buffer.from(charArr.buffer).toString('base64');
        }

        {
            let kerningArr = new Int16Array(fontSrc.kernings.length * 3);
            idx = 0;
            let nonZeroCount = 0;

            for (let k of fontSrc.kernings) {
                if (k.amount === 0) {
                    continue;
                }
                nonZeroCount++;
                for (let x of [k.first, k.second, k.amount]) {
                    kerningArr[idx++] = x;
                }
            }
            kerningArr = kerningArr.slice(0, nonZeroCount * 3);

            face.kernings = Buffer.from(kerningArr.buffer).toString('base64');
        }

        console.log(`Processing face: ${name}, total characters: ${fontSrc.chars.length}`);
        fontData.faces.push(face);
    }

    let result = JSON.stringify(fontData);

    fs.mkdirSync('public/fonts', { recursive: true });
    fs.writeFileSync('public/fonts/font-def.json', result, { encoding: 'utf8' });
    fs.copyFileSync('fonts/font-atlas.png', 'public/fonts/font-atlas.png');
}


runAll().then();
