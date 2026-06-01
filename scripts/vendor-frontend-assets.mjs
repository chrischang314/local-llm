import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const assetRoot = path.join(rootDir, "frontend", "vendor");

const assets = [
  {
    from: "node_modules/marked/marked.min.js",
    to: "marked/4.3.0/marked.min.js",
  },
  {
    from: "node_modules/dompurify/dist/purify.min.js",
    to: "dompurify/3.4.7/purify.min.js",
  },
  {
    from: "node_modules/@highlightjs/cdn-assets/highlight.min.js",
    to: "highlight.js/11.10.0/highlight.min.js",
  },
  {
    from: "node_modules/@highlightjs/cdn-assets/styles/github-dark.min.css",
    to: "highlight.js/11.10.0/styles/github-dark.min.css",
  },
  {
    from: "node_modules/lucide/dist/umd/lucide.min.js",
    to: "lucide/0.468.0/lucide.min.js",
  },
];

for (const asset of assets) {
  const source = path.join(rootDir, ...asset.from.split("/"));
  const destination = path.join(assetRoot, ...asset.to.split("/"));
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
}

console.log(`Copied ${assets.length} frontend assets to ${path.relative(rootDir, assetRoot)}`);
