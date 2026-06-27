/**
 * Registration flow: start the camera, let the person frame their face,
 * capture one still frame, and stash it in a hidden form field before submit.
 */
document.addEventListener("DOMContentLoaded", async () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const viewfinder = document.getElementById("viewfinder");
  const captureBtn = document.getElementById("captureBtn");
  const retakeBtn = document.getElementById("retakeBtn");
  const submitBtn = document.getElementById("submitBtn");
  const statusLine = document.getElementById("statusLine");
  const imageDataInput = document.getElementById("image_data");
  const placeholder = document.getElementById("placeholder");

  const BURST_COUNT = 5;
  const BURST_INTERVAL_MS = 350;
  let captured = false;

  function setStatus(text, kind) {
    statusLine.textContent = text;
    statusLine.className = "status-line" + (kind ? " " + kind : "");
  }

  try {
    await window.Camera.start(video);
    placeholder.style.display = "none";
    viewfinder.classList.add("scanning");
    setStatus("Camera ready — hold still and capture.");
  } catch (err) {
    setStatus("Camera unavailable. Allow camera access and reload.", "err");
    captureBtn.disabled = true;
    return;
  }

  captureBtn.addEventListener("click", async () => {
    captureBtn.disabled = true;
    const frames = [];

    // Capture a short burst of frames instead of one still image — this gives
    // the LBPH recognizer several angles/lighting moments to train on, which
    // noticeably improves login accuracy versus a single snapshot.
    for (let i = 0; i < BURST_COUNT; i++) {
      frames.push(window.Camera.captureFrame(video, canvas));
      setStatus(`Capturing… ${i + 1}/${BURST_COUNT} (hold still)`);
      await new Promise((resolve) => setTimeout(resolve, BURST_INTERVAL_MS));
    }

    imageDataInput.value = JSON.stringify(frames);
    captured = true;

    canvas.style.display = "block";
    video.style.display = "none";
    viewfinder.classList.remove("scanning");

    captureBtn.style.display = "none";
    retakeBtn.style.display = "inline-flex";
    submitBtn.disabled = false;
    setStatus(`Captured ${BURST_COUNT} frames. Submit to finish registering.`, "ok");
  });

  retakeBtn.addEventListener("click", () => {
    canvas.style.display = "none";
    video.style.display = "block";
    viewfinder.classList.add("scanning");

    captureBtn.style.display = "inline-flex";
    captureBtn.disabled = false;
    retakeBtn.style.display = "none";
    submitBtn.disabled = true;
    imageDataInput.value = "";
    captured = false;
    setStatus("Camera ready — hold still and capture.");
  });

  document.getElementById("registerForm").addEventListener("submit", (e) => {
    if (!captured) {
      e.preventDefault();
      setStatus("Capture your face before submitting.", "err");
    }
  });
});
