# Vendored Frontend Assets

These files are served from the same origin as the Local LLM frontend so the
LAN UI does not depend on public CDNs at runtime.

Pinned sources:

- `marked-4.3.0.min.js` from `marked@4.3.0`
- `dompurify-3.4.7.min.js` from `dompurify@3.4.7`
- `highlight-11.10.0.min.js` from Highlight.js CDN release `11.10.0`
- `highlight-github-dark-11.10.0.min.css` from Highlight.js CDN release `11.10.0`
- `lucide-1.17.0.min.js` from `lucide@1.17.0`

When updating these assets, change the filenames, HTML references, and
`tests/frontend-health-status.test.mjs` together so runtime HTML cannot drift
back to unpinned CDN URLs.
