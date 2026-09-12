/* =========================================================
 * 官方 MediaPipe FaceLandmarker 封装(替代手拼 ONNX 识别)
 * 返回 468 个归一化 [0..1] 关键点, 供页面画点 + 评分
 * ========================================================= */
(function () {
  'use strict';
  var S = { ready: false, error: null, face: null, last: null };
  var pending = null;

  function doInit(wasmBase) {
    return (async function () {
      var m = await import('/mp/vision_bundle.mjs');   // 绝对路径(相对页面根)
      var vision = await m.FilesetResolver.forVisionTasks(wasmBase);
      var face = await m.FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/mp/face_landmarker.task', delegate: 'CPU' },
        runningMode: 'IMAGE',
        numFaces: 1,
        outputFacialTransformationMatrixes: false,
        outputFaceBlendshapes: false,
      });
      S.face = face; S.ready = true; S.error = null;
      return true;
    })();
  }

  // 优先本地 mp/(WebView 自包含), 失败回退官方 CDN
  function load() {
    if (pending) return pending;
    pending = (async function () {
      try { await doInit('/mp'); console.log('[facemesh] 本地 wasm 加载成功'); return; }
      catch (e1) {
        console.warn('[facemesh] 本地失败, 转 CDN:', e1 && e1.message);
        try { await doInit('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'); return; }
        catch (e2) {
          S.error = (e2 && e2.message) || String(e2);
          console.error('[facemesh] 加载失败:', e2);
        }
      }
    })();
  }

  function detect(src) {
    if (!S.ready) return null;
    try {
      var result = S.face.detect(src);
      if (result && result.faceLandmarks && result.faceLandmarks.length) {
        var lm = result.faceLandmarks[0];
        var pts = [];
        for (var i = 0; i < lm.length; i++) pts.push([lm[i].x, lm[i].y]);
        S.last = { pts: pts, faces: result.faceLandmarks.length };
        return S.last;
      }
      return null;
    } catch (e) { S.lastErr = (e && e.message) || String(e); return null; }
  }

  window.__FM = {
    load: load,
    detect: detect,
    isReady: function () { return S.ready; },
    getError: function () { return S.error; },
    last: function () { return S.last; },
  };
})();