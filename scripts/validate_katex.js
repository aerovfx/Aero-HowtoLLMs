#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const katex = require('markdown-it-katex');

const ROOT = path.resolve(__dirname, '..');
const DOCS = path.join(ROOT, 'docs');

function walk(dir) {
  let results = [];
  const list = fs.readdirSync(dir, { withFileTypes: true });
  for (const ent of list) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      results = results.concat(walk(p));
    } else if (ent.isFile() && p.endsWith('.md')) {
      results.push(p);
    }
  }
  return results;
}

function run() {
  const md = new MarkdownIt({ html: true })
    .use(katex, { throwOnError: true, errorColor: '#cc0000' });

  const files = [];
  // top-level README.md
  const readme = path.join(ROOT, 'README.md');
  if (fs.existsSync(readme)) files.push(readme);
  if (fs.existsSync(DOCS)) files.push(...walk(DOCS));

  let errors = [];
  for (const f of files) {
    try {
      const src = fs.readFileSync(f, 'utf8');
      // render; if any KaTeX syntax is invalid, plugin will throw
      md.render(src);
    } catch (e) {
      errors.push({ file: f, error: e && e.message ? e.message : String(e) });
    }
  }

  if (errors.length === 0) {
    console.log('KaTeX validation passed for all scanned Markdown files.');
    process.exit(0);
  } else {
    console.log('KaTeX validation found errors:');
    for (const err of errors) {
      console.log(`- ${path.relative(ROOT, err.file)}: ${err.error}`);
    }
    process.exit(2);
  }
}

run();
