/**
 * Shared webcam helpers used by both the registration and login pages.
 * Exposes `Camera` on window with start/stop/captureFrame.
 */
window.Camera = (function () {
  let stream = null;

  async function start(videoEl) {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 360, facingMode: "user" },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    return stream;
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
  }

  /** Grabs the current video frame and returns a base64 data URL (image/jpeg). */
  function captureFrame(videoEl, canvasEl) {
    const w = videoEl.videoWidth || 480;
    const h = videoEl.videoHeight || 360;
    canvasEl.width = w;
    canvasEl.height = h;
    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, w, h);
    return canvasEl.toDataURL("image/jpeg", 0.85);
  }

  return { start, stop, captureFrame };
})();
