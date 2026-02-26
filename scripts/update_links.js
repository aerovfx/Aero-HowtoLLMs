// update_links.js
// This script scans all .md files under the docs directory and ensures that markdown links point to existing .md files.
// It rewrites links to use correct relative paths.

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

function normalizeLink(link, fromFile) {
    // If link is an absolute URL or starts with http, leave unchanged
    if (/^(http|https):\/\//.test(link)) return link;
    // Resolve the link relative to the source file
    const fromDir = path.dirname(fromFile);
    const targetPath = path.resolve(fromDir, link);
    if (fs.existsSync(targetPath)) {
        // Compute relative path from source file to target
        let rel = path.relative(fromDir, targetPath);
        // Use POSIX style for markdown
        rel = rel.split(path.sep).join('/');
        return rel;
    }
    // If file does not exist, keep original (could be a heading anchor)
    return link;
}

function processFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    let changed = false;
    content = content.replace(linkRegex, (match, text, link) => {
        const newLink = normalizeLink(link, filePath);
        if (newLink !== link) {
            changed = true;
            return `[${text}](${newLink})`;
        }
        return match;
    });
    if (changed) {
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`Updated links in ${filePath}`);
    }
}

function main() {
    const mdFiles = getAllMdFiles(docsDir);
    mdFiles.forEach(processFile);
}

main();
