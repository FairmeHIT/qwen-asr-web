const form = document.querySelector("#form");
const fileInput = document.querySelector("#file");
const fileMeta = document.querySelector("#fileMeta");
const output = document.querySelector("#output");
const statusEl = document.querySelector("#status");
const submit = document.querySelector("#submit");
const summarize = document.querySelector("#summarize");
const downloadText = document.querySelector("#downloadText");
const downloadJson = document.querySelector("#downloadJson");
const downloadSummary = document.querySelector("#downloadSummary");
const downloadSummaryJson = document.querySelector("#downloadSummaryJson");
const summaryInstruction = document.querySelector("#summaryInstruction");
const summaryOutput = document.querySelector("#summaryOutput");

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
    if (!health.llm_configured) {
      summaryOutput.placeholder = "未配置 LLM_API_KEY 或 DEEPSEEK_API_KEY，配置后可提炼要点";
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
  setDownload(downloadSummary, "");
  setDownload(downloadSummaryJson, "");
  submit.disabled = true;
  submit.querySelector("span").textContent = "转写中";
  output.value = "";
  summaryOutput.value = "";
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

summarize.addEventListener("click", async () => {
  const text = output.value.trim();
  if (!text) {
    setStatus("没有可提炼文本", "error");
    return;
  }

  setDownload(downloadSummary, "");
  setDownload(downloadSummaryJson, "");
  summarize.disabled = true;
  summarize.textContent = "提炼中";
  summaryOutput.value = "";
  setStatus("要点提炼中", "ready");

  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        instruction: summaryInstruction.value || "",
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Summary failed");
    }
    summaryOutput.value = data.summary || "";
    setDownload(downloadSummary, data.summary_url);
    setDownload(downloadSummaryJson, data.json_url);
    setStatus(`提炼完成 · ${data.model}`, "ready");
  } catch (error) {
    summaryOutput.value = error.message || String(error);
    setStatus("提炼失败", "error");
  } finally {
    summarize.disabled = false;
    summarize.textContent = "提炼要点";
  }
});

loadHealth();
