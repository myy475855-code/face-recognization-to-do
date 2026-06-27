/**
 * Login flow: continuously capture frames and POST them to /api/face-login
 * until a match is found (or the person switches to the password fallback).
 */
document.addEventListener("DOMContentLoaded", async () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const viewfinder = document.getElementById("viewfinder");
  const statusLine = document.getElementById("statusLine");
  const placeholder = document.getElementById("placeholder");
  const passwordToggle = document.getElementById("passwordToggle");
  const passwordPanel = document.getElementById("passwordPanel");

  const SCAN_INTERVAL_MS = 1600;
  let scanning = false;
  let timer = null;

  function setStatus(text, kind) {
    statusLine.textContent = text;
    statusLine.className = "status-line" + (kind ? " " + kind : "");
  }

  async function scanOnce() {
    if (!scanning) return;
    const dataUrl = window.Camera.captureFrame(video, canvas);

    try {
      const res = await fetch("/api/face-login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: dataUrl }),
      });
      const result = await res.json();

      if (result.success) {
        scanning = false;
        viewfinder.classList.remove("scanning");
        setStatus(result.message, "ok");
        window.Camera.stop();
        window.location.href = result.redirect;
        return;
      }

      setStatus(result.message || "Scanning…");
    } catch (err) {
      setStatus("Connection issue — retrying…", "err");
    }

    if (scanning) timer = setTimeout(scanOnce, SCAN_INTERVAL_MS);
  }

  try {
    await window.Camera.start(video);
    placeholder.style.display = "none";
    viewfinder.classList.add("scanning");
    scanning = true;
    setStatus("Scanning for a registered face…");
    scanOnce();
  } catch (err) {
    setStatus("Camera unavailable — use your password instead.", "err");
    passwordPanel.style.display = "block";
  }

  passwordToggle.addEventListener("click", () => {
    const showing = passwordPanel.style.display === "block";
    passwordPanel.style.display = showing ? "none" : "block";
    passwordToggle.textContent = showing ? "Use password instead" : "Hide password login";
  });

  window.addEventListener("beforeunload", () => {
    scanning = false;
    clearTimeout(timer);
    window.Camera.stop();
  });
});
