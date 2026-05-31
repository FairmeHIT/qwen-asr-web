const form = document.querySelector("#form");
const fileInput = document.querySelector("#file");
const fileMeta = document.querySelector("#fileMeta");
const output = document.querySelector("#output");
const statusEl = document.querySelector("#status");
const submit = document.querySelector("#submit");
const downloadText = document.querySelector("#downloadText");
const downloadJson = document.querySelector("#downloadJson");

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`.trim();
}

function setDownload(link, url) {
  if (!url) {
    link.href = "#";
    link.classList.add("disabled");
    return;
  }
  link.href = url;
  link.classList.remove("disabled");
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const health = await res.json();
    if (health.cuda_available) {
      setStatus(`${health.gpu || "CUDA"} ready`, "ready");
    } else {
      setStatus("CUDA 未可用", "error");
    }
  } catch (error) {
    setStatus("服务未就绪", "error");
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    fileMeta.textContent = "支持长音频；视频文件需要系统安装 ffmpeg";
    return;
  }
  const sizeMb = file.size / 1024 / 1024;
  fileMeta.textContent = `${file.name} · ${sizeMb.toFixed(1)} MB`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus("请选择文件", "error");
    return;
  }

  setDownload(downloadText, "");
  setDownload(downloadJson, "");
  submit.disabled = true;
  submit.querySelector("span").textContent = "转写中";
  output.value = "";
  setStatus("模型处理中", "ready");

  try {
    const formData = new FormData(form);
    const res = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Transcription failed");
    }
    output.value = data.text || "";
    setDownload(downloadText, data.text_url);
    setDownload(downloadJson, data.json_url);
    setStatus(data.language ? `完成 · ${data.language}` : "完成", "ready");
  } catch (error) {
    output.value = error.message || String(error);
    setStatus("转写失败", "error");
  } finally {
    submit.disabled = false;
    submit.querySelector("span").textContent = "开始转写";
  }
});

loadHealth();
