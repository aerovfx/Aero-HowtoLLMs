// check_links.js
// This script scans all .md files under the docs directory and reports markdown links that point to non‑existent .md files.

const fs = require('fs');
const path = require('path');

const docsDir = path.resolve(__dirname, '..', 'docs');

function getAllMdFiles(dir) {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(file => {
        const fullPath = path.join(dir, file);
        const stat = fs.statSync(fullPath);
        if (stat && stat.isDirectory()) {
            results = results.concat(getAllMdFiles(fullPath));
        } else if (fullPath.endsWith('.md')) {
            results.push(fullPath);
        }
    });
    return results;
}

function resolveLink(link, fromFile) {
    if (/^(http|https):\/\//.test(link)) return null; // external link, ignore
    const fromDir = path.dirname(fromFile);
    const targetPath = path.resolve(fromDir, link);
    return targetPath;
}

function processFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    let match;
    while ((match = linkRegex.exec(content)) !== null) {
        const [, text, link] = match;
        if (!link.endsWith('.md')) continue; // only care about .md links
        const target = resolveLink(link, filePath);
        if (target && !fs.existsSync(target)) {
            console.log(`Broken link in ${filePath}: [${text}](${link}) -> ${target} does not exist`);
        }
    }
}

function main() {
    const mdFiles = getAllMdFiles(docsDir);
    mdFiles.forEach(processFile);
}

main();
