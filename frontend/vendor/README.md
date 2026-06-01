# Frontend Vendor Assets

These browser assets are copied from pinned npm packages so the LAN UI does not
depend on public CDNs at runtime.

Refresh after dependency updates:

```powershell
npm install
npm run vendor:frontend
```

Pinned assets:

- Marked `4.3.0`, MIT, `marked/4.3.0/marked.min.js`
- DOMPurify `3.4.7`, MPL-2.0 OR Apache-2.0, `dompurify/3.4.7/purify.min.js`
- Highlight.js CDN assets `11.10.0`, BSD-3-Clause,
  `highlight.js/11.10.0/highlight.min.js` and
  `highlight.js/11.10.0/styles/github-dark.min.css`
- Lucide `0.468.0`, ISC, `lucide/0.468.0/lucide.min.js`
