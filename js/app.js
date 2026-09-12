/* =========================================================
 * AI 颜值分 · Web 版 (onnxruntime-web + MediaPipe face_mesh ONNX)
 *
 * 模型: models/face_landmark.onnx
 *   输入 input_12  -> [1, 256, 256, 3]  RGB 像素值(float32)
 *   输出 Identity  -> [1,1,1,1434]      = 478 关键点, 每点 (x,y,z) 像素坐标
 *       Identity_1 -> score,  Identity_2 -> 3D 姿态
 *
 * 页面元素由 index.html 提供。逻辑：
 *   摄像头/选图 -> 缩放到 256x256 -> 推理 -> 得 478 点 -> 几何计分 -> 画点
 * ========================================================= */

(function () {
  'use strict';

  // ---------- DOM ----------
  var $ = function (id) { return document.getElementById(id); };
  var video = $('cam'), overlay = $('overlay'), hint = $('empty-hint');
  var scoreEl = $('score'), barEl = $('scorebar'), metricsEl = $('metrics');
  var btnCam = $('btn-camera'), btnPick = $('btn-pick'), fileInput = $('file');

  var oc = overlay.getContext('2d');
  var W = 360, H = 480;   // CSS 像素坐标(每次 resize 更新)
  function resizeCanvas() {
    var cw = overlay.clientWidth || overlay.parentElement.clientWidth || 360;
    var ch = overlay.clientHeight || overlay.parentElement.clientHeight || 480;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(160, cw); H = Math.max(160, ch);
    overlay.width = Math.round(W * dpr);
    overlay.height = Math.round(H * dpr);
    oc.setTransform(dpr, 0, 0, dpr, 0, 0);   // 之后以 CSS 像素绘制
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ---------- 运行状态 ----------
  var S = {
    ort: null,
    session: null,
    inputName: 'input_12',
    outputName: 'Identity',
    inSize: 256,          // 模型输入边长 256
    pts: null,            // 478x2 归一化到 [0,1]
    running: false,
    engine: null,         // 'mp' | 'onnx' | null
    beingPhoto: false,    // 是否处于选图模式
    source: null,         // ImageData/VideoFrame
    bgImage: null,        // 选图模式下要作为背景显示的照片
    bgNaturalW: 0,
    bgNaturalH: 0,
  };

  // ---------- 日志 ----------
  // 默认只进控制台; 保存到内存缓冲, 供 debug 面板查看(不主动显示到页面)
  var debugLogs = [];    // [{t:时间, msg, err}]
  function log(msg, isErr) {
    var d = new Date();
    var t = (d.getHours()+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0'));
    debugLogs.push({ t: t, msg: String(msg), err: !!isErr });
    if (debugLogs.length > 300) debugLogs.shift();   // 限制条数
    console.log((isErr?'[err] ':'[log] ')+msg);
  }
  // 全局入口, 供页面错误捕获等调用(需挂到 window, 因为 IIFE 内)
  window.__log = log;

  function renderDebugPanel() {
    var body = $('dbg-body');
    if (!body) return;
    body.innerHTML = '';
    debugLogs.forEach(function (e) {
      var el = document.createElement('div');
      el.className = 'dbg-line' + (e.err ? ' err' : '');
      el.textContent = '[' + e.t + '] ' + e.msg;
      body.appendChild(el);
    });
    body.scrollTop = body.scrollHeight;   // 置底
  }
  function toggleDebug(force) {
    var ov = $('debug');
    if (!ov) return;
    var wasHidden = ov.classList.contains('hidden');
    var open;
    if (force === true) open = true;
    else if (force === false) open = false;
    else open = wasHidden;          // 未指定 force = 切开关
    ov.classList.toggle('hidden', !open);
    if (open) renderDebugPanel();
  }
  var dbgBtn = $('btn-debug');
  if (dbgBtn) dbgBtn.addEventListener('click', function(){ toggleDebug(true); });
  var dbgClose = $('dbg-close'); if (dbgClose) dbgClose.addEventListener('click', function () { toggleDebug(false); });
  var dbgClear = $('dbg-clear'); if (dbgClear) dbgClear.addEventListener('click', function () { debugLogs.length = 0; if(!$('debug').classList.contains('hidden')) renderDebugPanel(); });
  function setDot(cls, txt) {
    var dot = $('status-dot');
    dot.className = 'dot ' + cls;
    dot.title = txt;
    var st = $('status-text');
    if (st) st.textContent = txt;   // 右上角显示状态文字(截断显示)
    if (txt) log(txt);
  }

  // ---------- 加载模型(浏览器 fetch 模型文件) ----------
  async function loadModel() {
    var ort = window.ort || window.onnxruntime || window.onnx;
    if (!ort || !ort.InferenceSession) {
      // 兼容 1.20 UMD: 挂载 window.ort
      ort = window.ort || window.onnx;
      if (!ort || !ort.InferenceSession) throw new Error('onnxruntime 未加载');
    }
    S.model = ort;
    // 指定 wasm 位置 + 强制单线程(避免 worker 依赖, 更稳)
    if (ort.env && ort.env.wasm) {
      ort.env.wasm.wasmPaths = '/ort/';        // 绝对根路径 => 新 URL('ory.js', '/ort/')=  /ort/ort-wasm...
      if (ort.env.wasm.numThreads > 1) ort.env.wasm.numThreads = 1;
    }
    setDot('loading', '加载模型 models/face_landmark.onnx ...');
    var send = await ort.InferenceSession.create('models/face_landmark.onnx');
    S.session = send;
    // 读取真实输入名/形状(自适应)
    if (send.inputMetadata && send.inputMetadata[S.inputName]) {
      var d = send.inputMetadata[S.inputName].shape;
      if (d && d.length >= 4 && d[2]) S.inSize = d[2];   // 256
    }
    setDot('ready', '模型就绪 ✓ (' + S.inSize + 'x' + S.inSize + ')');
    log('模型就绪, 输入尺寸 ' + S.inSize + ', 输出 ' + send.outputNames.join(','));
  }

  // ---------- 把 source(ImageBitmap/Video 帧) 缩放到模型尺寸, 得 Float32 张量 ----------
  function makeInput(src) {
    var n = S.inSize;
    // 绘到临时 256x256 canvas(RGB)
    var tc = document.createElement('canvas'); tc.width = n; tc.height = n;
    var tctx = tc.getContext('2d');
    tctx.fillStyle = '#000'; tctx.fillRect(0, 0, n, n);
    var sw = src.videoWidth || src.naturalWidth || src.width || 0;
    var sh = src.videoHeight || src.naturalHeight || src.height || 0;
    var aspect = (sw && sh) ? (sw / sh) : 1;
    var cw = n, ch = n;
    if (aspect > 1) { cw = n; ch = n / aspect; } else { ch = n; cw = n * aspect; }
    tctx.drawImage(src, (n - cw) / 2, (n - ch) / 2, cw, ch);
    // 像素读取
    var id = tctx.getImageData(0, 0, n, n);
    var data = id.data;
    // 转成 [1,n,n,3] Float32, 值域保持 0-255 (模型输出也是此尺度)
    var f32 = new Float32Array(n * n * 3);
    var ci = 0;
    for (var i = 0; i < n * n; i++) {
      f32[ci++] = data[i * 4];
      f32[ci++] = data[i * 4 + 1];
      f32[ci++] = data[i * 4 + 2];
    }
    return new S.model.Tensor('float32', f32, [1, n, n, 3]);
  }

  // ---------- 推理 + 解析关键点 ----------
  // 共享的"磨出点 → 算分 → 画点"流程(两套引擎共用)
  function finalizeWithPoints(pts) {
    if (!pts || !pts.length) {
      _clearFace(true);
      return;
    }
    var real = pts.filter(function (q) { return q[0] >= -0.2 && q[0] <= 1.2 && q[1] >= -0.2 && q[1] <= 1.2; });
    var use = real.length >= 3 ? real : pts;
    var minX = Math.min.apply(null, use.map(function (q) { return q[0]; }));
    var maxX = Math.max.apply(null, use.map(function (q) { return q[0]; }));
    var minY = Math.min.apply(null, use.map(function (q) { return q[1]; }));
    var maxY = Math.max.apply(null, use.map(function (q) { return q[1]; }));
    S.pts = use;
    S.faceBox = { minX: minX, maxX: maxX, minY: minY, maxY: maxY };
    var sc = scoreFromPoints(use);
    showScore(sc);
    drawKeypoints(use);
    if (!S.running) log('检测到人脸: ' + use.length + ' 关键点, 分 ' + sc.score);
  }
  function _clearFace(updateUI) {
    S.pts = null; S.faceBox = null; S._vw = null;
    if (updateUI) {
      scoreEl.textContent = '--';
      barEl.value = 0;
      metricsEl.innerHTML = '';
      _drawNoFace();
    }
  }
  // 未识别到人脸时: 保留照片(整图 contain)并提示, 避免黑屏
  function _drawNoFace(msg) {
    if (S.bgImage) {
      var bw = S.bgNaturalW || 1, bh = S.bgNaturalH || 1;
      var fsc = Math.max(W / bw, H / bh);
      oc.save();
      try { oc.filter = 'blur(14px) brightness(.55)'; } catch (e) {}
      oc.drawImage(S.bgImage, (W - bw * fsc) / 2, (H - bh * fsc) / 2, bw * fsc, bh * fsc);
      oc.restore();
      var scale = Math.min(W / bw, H / bh);
      var ox = (W - bw * scale) / 2, oy = (H - bh * scale) / 2;
      oc.drawImage(S.bgImage, ox, oy, bw * scale, bh * scale);
    } else {
      oc.clearRect(0, 0, overlay.width, overlay.height);
    }
    // 提示横幅
    var msg2 = msg || '未识别到人脸，请换张正脸照';
    var pad = 16, fw = 15, th = 44;
    var tw = Math.min(W - pad * 2, oc.measureText ? (oc.font = fw + 'px system-ui,sans-serif', oc.measureText(msg2).width + 40) : 300);
    var hw = tw, sx = (W - tw) / 2, sy = 24;
    oc.fillStyle = 'rgba(0,0,0,.55)';
    oc.beginPath();
    if (oc.roundRect) oc.roundRect(sx, sy, tw, th, th / 2); else oc.rect(sx, sy, tw, th);
    oc.fill();
    oc.fillStyle = '#ffd54f';
    oc.font = 'bold ' + fw + 'px system-ui, sans-serif';
    oc.textAlign = 'center'; oc.textBaseline = 'middle';
    oc.fillText(msg2, W / 2, sy + th / 2);
  }

  // 超大图先降采样到上限边长(提速 + 防内存 + 避免 MediaPipe 漏检)
  function _cappedSource(src) {
    var sw = src.videoWidth || src.naturalWidth || src.width || 0;
    var sh = src.videoHeight || src.naturalHeight || src.height || 0;
    var CAP = 1200;   // 最长边上限
    if (!sw || !sh || (sw <= CAP && sh <= CAP)) return src;
    var sc = Math.min(CAP / sw, CAP / sh);
    var w = Math.max(1, Math.round(sw * sc)), h = Math.max(1, Math.round(sh * sc));
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    c.getContext('2d').drawImage(src, 0, 0, w, h);
    return c;
  }

  async function inferAndScore(src) {
    // 首选官方 MediaPipe FaceLandmarker(主引擎)
    if (window.__FM && window.__FM.isReady()) {
      var det = _cappedSource(src);
      var fm = window.__FM.detect(det);
      if (fm && fm.pts && fm.pts.length >= 3) { S.engine = 'mp'; finalizeWithPoints(fm.pts); return; }
      // 未检测到人脸
      _clearFace(true);
      if (!S.running && S.beingPhoto) log('未检测到人脸，请换张正脸照', false);
      return;
    }

    // 回退: 自研 ONNX(face_landmark.onnx) 路径
    if (!S.session) return;
    var inputTensor = makeInput(src);
    var feeds = {}; feeds[S.inputName] = inputTensor;
    var out;
    try { out = await S.session.run(feeds); }
    catch (e) { log('推理失败: ' + (e && e.message), true); return; }

    var lm = null;
    for (var k in out) { var t = out[k]; if (t && t.data && t.data.length === 1434) { lm = t.data; break; } }
    if (!lm) { for (var k2 in out) { var t2 = out[k2]; if (t2 && t2.data && t2.data.length % 3 === 0 && t2.data.length > 100) { lm = t2.data; break; } } }
    if (!lm) { log('未找到关键点输出', true); return; }

    var N = lm.length / 3, orig = S.inSize;
    var pts2 = [];
    for (var p = 0; p < N; p++) { pts2.push([lm[p * 3] / orig, lm[p * 3 + 1] / orig]); }
    // ONNX 输出假设为 0..原尺寸, 快速检查合理性(若全部异常, 忽略并报错)
    if (pts2.every(function (q) { return q[0] < -1 || q[0] > 2 || q[1] < -1 || q[1] > 2; })) {
      _clearFace(true);
      if (!S.running) log('ONNX 关键点越界，识别无效（请改用 MediaPipe 引擎）', true);
      return;
    }
    S.engine = 'onnx';
    finalizeWithPoints(pts2);
  }

  // ---------- 几何计分 (基于478点拓扑规则的简化几何特征) ----------
  function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
  function scoreFromPoints(pts) {
    var b = S.faceBox;
    var faceW = b.maxX - b.minX;
    var faceH = b.maxY - b.minY;
    if (faceW <= 0 || faceH <= 0) return { score: 50, ratioScore: 0, symScore: 0, thirdScore: 0 };

    // 1) 面部高宽比接近黄金比 0.72(最好 0.72~0.82)
    var ratio = faceW / faceH;
    var ratioScore = clamp(100 - Math.abs(ratio - 0.76) * 220, 20, 100);

    // 2) 左右对称性: 以 x 中线为轴, 比较左/右点分布
    var midX = (b.minX + b.maxX) / 2;
    var left = 0, right = 0;
    for (var i = 0; i < pts.length; i++) { (pts[i][0] <= midX ? left++ : right++); }
    var asym = Math.abs(left - right) / Math.max(1, (left + right));
    var symScore = Math.max(20, 100 - asym * 180);

    // 3) 竖向比例(三停)：鼻子在脸中部时最佳
    var noseY = midYOfNose(pts);
    var thirdScore = Math.max(20, 100 - Math.abs(noseY - 0.5) * 60);

    var score = Math.round(ratioScore * 0.35 + symScore * 0.35 + thirdScore * 0.30);
    score = Math.max(20, Math.min(98, score));

    return {
      score: score,
      ratioScore: Math.round(ratioScore),
      symScore: Math.round(symScore),
      thirdScore: Math.round(thirdScore),
    };
  }
  function midYOfNose(pts) {
    // 用下半部点找出最接近脸部纵向中心处集中成为"鼻子"估点(简化:取 y 坐标中位数附近密集区)
    var ys = pts.map(function (q) { return q[1]; }).sort(function (a, b) { return a - b; });
    return ys[Math.floor(ys.length * 0.5)] || 0.5;
  }

  function showScore(sc) {
    scoreEl.textContent = sc.score;
    barEl.value = sc.score;
    metricsEl.innerHTML = '';
    addMetric('脸宽/高比', sc.ratioScore, '接近黄金比 0.72');
    addMetric('左右对称', sc.symScore, '以中线分布计');
    addMetric('纵向三庭', sc.thirdScore, '鼻在脸高 0.5 处');
    addMetric('综合', sc.score, '几何特征加权');
  }
  function addMetric(label, v, note) {
    var d = document.createElement('div');
    d.className = 'metric';
    d.innerHTML = label + '<br/><b>' + v + '</b><span style="color:#7d8bb0"> ' + note + '</span>';
    metricsEl.appendChild(d);
  }

  // ---------- 绘制关键点 ----------
  function drawKeypoints(pts) {
    oc.clearRect(0, 0, W, H);
    // 全局"视图"把原图[归一化 0..1]坐标映射到画布CSS像素
    function sx(f) { var V = S._vw; return V ? V.ox + f * V.bw * V.scale : f * W; }
    function sy(f) { var V = S._vw; return V ? V.oy + f * V.bh * V.scale : f * H; }

    S._vw = null;
    if (S.bgImage) {
      var bw = S.bgNaturalW || 1, bh = S.bgNaturalH || 1;
      var scale, ox, oy;
      if (S.faceBox && S.faceBox.maxX > S.faceBox.minX) {
        // 有人脸: 以人脸为中心放大(cover裁切, 不拉伸), 脸约占画面高 62%、中间偏上
        var fcx = (S.faceBox.minX + S.faceBox.maxX) / 2;
        var fcy = (S.faceBox.minY + S.faceBox.maxY) / 2;
        var fwpx = Math.max(1, (S.faceBox.maxX - S.faceBox.minX) * bw);
        var fhpx = Math.max(1, (S.faceBox.maxY - S.faceBox.minY) * bh);
        var tW = Math.max(40, W * 0.8), tH = Math.max(40, H * 0.68);
        scale = Math.min(tW / fwpx, tH / fhpx) * 1.15;
        ox = W / 2 - fcx * bw * scale;
        oy = H / 2 - fcy * bh * scale;
      } else {
        // 无脸: 整图 contain 居中
        scale = Math.min(W / bw, H / bh);
        ox = (W - bw * scale) / 2;
        oy = (H - bh * scale) / 2;
      }
      // 1) 虚化放大图铺满(四边遮罩)
      var fsc = Math.max(W / bw, H / bh);
      oc.save();
      try { oc.filter = 'blur(14px) brightness(.55)'; } catch (e) {}
      oc.drawImage(S.bgImage, (W - bw * fsc) / 2, (H - bh * fsc) / 2, bw * fsc, bh * fsc);
      oc.restore();
      // 2) 主图: 人脸居中放大 / 整图 contain, 均不拉伸
      oc.drawImage(S.bgImage, ox, oy, bw * scale, bh * scale);
      S._vw = { bw: bw, bh: bh, scale: scale, ox: ox, oy: oy };
    }

    oc.fillStyle = 'rgba(104,214,255,0.9)';
    for (var i = 0; i < pts.length; i++) {
      var x = sx(pts[i][0]), y = sy(pts[i][1]);
      if (isFinite(x) && isFinite(y) && x > -2 && x < W + 2 && y > -2 && y < H + 2) oc.fillRect(x - 1, y - 1, 2.4, 2.4);
    }
    if (S.faceBox) {
      oc.strokeStyle = 'rgba(255,200,90,.6)';
      oc.strokeRect(sx(S.faceBox.minX), sy(S.faceBox.minY), sx(S.faceBox.maxX) - sx(S.faceBox.minX), sy(S.faceBox.maxY) - sy(S.faceBox.minY));
    }
  }

  // ---------- 持续从 <video> 取样推理 ----------
  function loop() {
    if (S.running && video.srcObject && video.readyState >= 2) {
      inferAndScore(snapshotCam());   // 用"镜像+cover"快照识别, 使关键点与所见画面完全对齐
    }
    requestAnimationFrame(loop);
  }

  // 摄像头画面是"镜像+cover(裁边铺满)"显示; 采样一个与屏幕所见完全一致的帧,
  // 让识别出的关键点归一坐标 = 显示坐标, 不因 cover 裁剪或镜像而偏移。
  function snapshotCam() {
    var vw = video.videoWidth || 640, vh = video.videoHeight || 640;
    var fsc = Math.max(W / vw, H / vh);        // 同 CSS cover
    var dw = vw * fsc, dh = vh * fsc;
    var c = document.createElement('canvas');
    c.width = Math.round(W); c.height = Math.round(H);
    var c8 = c.getContext('2d');
    c8.translate(W, 0); c8.scale(-1, 1);      // 镜像, 与 video transform 一致
    c8.drawImage(video, (W - dw) / 2, (H - dh) / 2, dw, dh);
    return c;
  }

  async function startCamera() {
    // 清除选图背景, 回到实时模式
    S.bgImage = null; S.bgNaturalW = 0; S.bgNaturalH = 0;
    S.beingPhoto = false;
    _clearFace(true);
    try {
      hint.classList.add('hide');
      var stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 640, facingMode: 'user' } });
      video.srcObject = stream;
      video.play().catch(function () {});
      // 等真的有画面帧再显示, 避免分理等待时出现"灰底+黑色控件条"的原生占位
      video.style.display = 'none';
      await Promise.race([
        new Promise(function (res) { var on = function () { video.removeEventListener('loadeddata', on); res(); }; video.addEventListener('loadeddata', on); }),
        new Promise(function (res) { setTimeout(res, 2500); })
      ]);
      video.style.display = 'block';
      S.running = true;
      setDot('ready', '摄像头就绪');
    } catch (e) {
      log('摄像头不可用: ' + (e && e.message), true);
    }
  }
  btnCam.addEventListener('click', startCamera);

  fileInput.addEventListener('change', function () {
    var f = fileInput.files && fileInput.files[0];
    if (!f) return;
    // 停止摄像头采集(若有), 切换到"选图"模式
    if (video.srcObject) { video.srcObject.getTracks().forEach(function (t) { t.stop(); }); video.srcObject = null; }
    S.running = false;
    S.beingPhoto = true;
    video.style.display = 'none';   // 照片模式下隐藏摄像头元素, 只留 canvas
    var url = URL.createObjectURL(f);
    var img = new Image();
    img.onload = function () {
      // 按用户要求: 显示用完整原图(不降采样、实际比例), 检测时内部再按需降采样
      S.bgImage = img; S.bgNaturalW = img.naturalWidth; S.bgNaturalH = img.naturalHeight;
      hint.classList.add('hide');
      setDot('ready', '已选图');
      inferAndScore(img);   // 检测用原始 img(推理内部会降采样控制内存)
    };
    img.onerror = function () { log('图片加载失败', true); };
    img.src = url;
  });
  btnPick.addEventListener('click', function () { fileInput.click(); });

  // ---------- 启动 ----------
  // modelReady 用显式标志, hook 在模型就绪后才给出, 自动化可 {轮询}
  var modelReady = false;
  (async function () {
    try {
      if (window.__FM) window.__FM.load();          // 启动官方 MediaPipe 引擎
      await loadModel();                             // ONNX 兜底
      modelReady = true;
      loop();
    } catch (e) {
      setDot('error', '模型加载失败: ' + (e && e.message));
      log('初始化失败: ' + (e && e.message), true);
    }
  })();

  // 调试/自动化挂钩(模型就绪后才真正可用)
  window.__faceTest = {
    ready: function () { return modelReady; },
    infer: function (src) { if (!modelReady) return Promise.resolve('not-ready'); return Promise.resolve(inferAndScore(src)); },
    setPhoto: function (imgOrCanvas) { S.bgImage = imgOrCanvas; S.bgNaturalW = imgOrCanvas.naturalWidth || imgOrCanvas.width || 0; S.bgNaturalH = imgOrCanvas.naturalHeight || imgOrCanvas.height || 0; },
    makeInput: makeInput,
    // 直接分析模型原始输出(诊断用): 不做过滤, 返回关键点输出的统计
    analyze: async function (src) {
      if (!modelReady) return 'not-ready';
      try {
        var t = makeInput(src);
        var feeds = {}; feeds[S.inputName] = t;
        var out = await S.session.run(feeds);
        // 找到 1434 或最长的 3 整除输出
        var lm = null;
        for (var k in out) { if (out[k] && out[k].data && out[k].data.length === 1434) { lm = out[k].data; break; } }
        if (!lm) for (var k2 in out) { if (out[k2] && out[k2].data && out[k2].data.length % 3 === 0 && out[k2].data.length > 100) { lm = out[k2].data; break; } }
        if (!lm) return 'no-large-output';
        var n = S.inSize, N = lm.length / 3;
        var minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9, inRange=0, total=0, zFirst=lm[2];
        for (var i = 0; i < N; i++) {
          var x = lm[i*3], y = lm[i*3+1];
          if (x < minx) minx = x; if (x > maxx) maxx = x;
          if (y < miny) miny = y; if (y > maxy) maxy = y;
          total++;
          if (x >= 0 && x <= n && y >= 0 && y <= n) inRange++;
          if (i === 0) zFirst = lm[2];
        }
        // 归一化后看中心 80% 点集宽高比, 判断是否像人脸(可救)
        var xs = [], ys = [];
        for (var j = 0; j < N; j++) { xs.push(lm[j*3]); ys.push(lm[j*3+1]); }
        var rangeX = (maxx - minx) || 1, rangeY = (maxy - miny) || 1;
        var nx = xs.map(function (v) { return (v - minx) / rangeX; }).sort(function (a, b) { return a - b; });
        var ny = ys.map(function (v) { return (v - miny) / rangeY; }).sort(function (a, b) { return a - b; });
        var lo = Math.floor(N * 0.1), hi = N - lo;
        var cW = nx[hi - 1] - nx[lo], cH2 = ny[hi - 1] - ny[lo];
        var aspect = +(cW / (cH2 || 1)).toFixed(2);
        return { N: N, len: lm.length, xmin: +minx.toFixed(2), xmax: +maxx.toFixed(2), ymin: +miny.toFixed(2), ymax: +maxy.toFixed(2), inRange: inRange, inFrac: +(inRange/total).toFixed(3), inSize: n, centralAspect: aspect, faceLike: (aspect >= 0.55 && aspect <= 1.4), Identity1: out['Identity_1'] ? out['Identity_1'].data[0] : null };
      } catch (e) { return 'analyze-err ' + (e && e.message); }
    },
    get: function () { var mpReady = !!(window.__FM && window.__FM.isReady && window.__FM.isReady()); return { pts: S.pts, score: scoreEl ? scoreEl.textContent : null, box: S.faceBox, engine: S.engine, mpReady: mpReady, session: !!S.session, inputName: S.inputName, inSize: S.inSize, ready: modelReady, bg: S.bgImage }; },
  };
})();