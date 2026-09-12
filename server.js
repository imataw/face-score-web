// 极简静态文件服务器(供本地打开 AI颜值分页面)
const http = require('http');
const fs = require('fs');
const path = require('path');
const root = __dirname;
const PORT = parseInt(process.argv[2] || '8097', 10);
const MIME = { '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.mjs':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.json':'application/json', '.onnx':'application/octet-stream', '.wasm':'application/wasm', '.png':'image/png', '.jpg':'image/jpeg', '.gif':'image/gif', '.svg':'image/svg+xml' };
http.createServer((req, res) => {
  let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  if (urlPath === '/') urlPath = '/index.html';
  const file = path.join(root, urlPath);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(file, (e, data) => {
    if (e) { res.writeHead(404); return res.end('not found: ' + urlPath); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Content-Length': data.length });
    res.end(data);
  });
}).listen(PORT, () => console.log('Web 颜值分服务已启动: http://127.0.0.1:' + PORT));