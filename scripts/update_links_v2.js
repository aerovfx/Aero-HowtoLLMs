// update_links_v2.js
// Improves markdown links in the docs folder.
// Replaces links to README.md with index.md (common case) and ensures .md links point to existing files.

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

function fixLink(link, fromFile) {
    // ignore external URLs
    if (/^(http|https):\/\//.test(link)) return link;
    // replace common README.md with index.md
    let newLink = link.replace(/README\.md$/i, 'index.md');
    // resolve the path to see if it exists
    const fromDir = path.dirname(fromFile);
    const targetPath = path.resolve(fromDir, newLink);
    if (fs.existsSync(targetPath)) {
        // compute relative path using POSIX separators
        let rel = path.relative(fromDir, targetPath).split(path.sep).join('/');
        return rel;
    }
    // if still not exists, keep original (will be reported later)
    return link;
}

function processFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    let changed = false;
    content = content.replace(linkRegex, (match, text, link) => {
        const updated = fixLink(link, filePath);
        if (updated !== link) {
            changed = true;
            return `[${text}](${updated})`;
        }
        return match;
    });
    if (changed) {
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`Fixed links in ${filePath}`);
    }
}

function main() {
    const mdFiles = getAllMdFiles(docsDir);
    mdFiles.forEach(processFile);
}

main();
