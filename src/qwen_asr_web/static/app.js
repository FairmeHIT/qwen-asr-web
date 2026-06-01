const form = document.querySelector("#form");
const fileInput = document.querySelector("#file");
const fileMeta = document.querySelector("#fileMeta");
const output = document.querySelector("#output");
const statusEl = document.querySelector("#status");
const submit = document.querySelector("#submit");
const summarize = document.querySelector("#summarize");
const jobMeta = document.querySelector("#jobMeta");
const logs = document.querySelector("#logs");
const progressBar = document.querySelector("#progressBar");
const downloadAudio = document.querySelector("#downloadAudio");
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

function errorMessage(data, fallback) {
  const detail = data && data.detail;
  if (!detail) {
    return fallback;
  }
  if (typeof detail === "string") {
    return detail;
  }
  if (detail.message) {
    return detail.message;
  }
  return JSON.stringify(detail, null, 2);
}

function setProgress(value) {
  const progress = Math.max(0, Math.min(Number(value) || 0, 100));
  progressBar.style.width = `${progress}%`;
}

function setLogs(items) {
  if (!items || !items.length) {
    logs.textContent = "暂无日志";
    return;
  }
  logs.textContent = items.join("\n");
  logs.scrollTop = logs.scrollHeight;
}

function appendInlineMarkdown(parent, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parent.append(document.createTextNode(text.slice(lastIndex, match.index)));
    }

    const token = match[0];
    if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    }
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    parent.append(document.createTextNode(text.slice(lastIndex)));
  }
}

function renderSummaryMarkdown(markdown) {
  const text = (markdown || "").trim();
  summaryOutput.replaceChildren();

  if (!text) {
    summaryOutput.classList.add("empty");
    summaryOutput.textContent = "点击“提炼要点”后，摘要会显示在这里";
    return;
  }

  summaryOutput.classList.remove("empty");
  let currentList = null;
  let currentListType = "";

  function closeList() {
    currentList = null;
    currentListType = "";
  }

  function appendBlock(element, content) {
    appendInlineMarkdown(element, content.trim());
    summaryOutput.append(element);
  }

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = String(Math.min(heading[1].length + 2, 4));
      appendBlock(document.createElement(`h${level}`), heading[2]);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      const type = bullet ? "ul" : "ol";
      if (!currentList || currentListType !== type) {
        closeList();
        currentList = document.createElement(type);
        currentListType = type;
        summaryOutput.append(currentList);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, bullet ? bullet[1] : numbered[1]);
      currentList.append(item);
      continue;
    }

    closeList();
    appendBlock(document.createElement("p"), line);
  }
}

function applyJobSnapshot(job) {
  jobMeta.textContent = `${job.status} · ${job.stage} · ${job.progress}%`;
  setProgress(job.progress);
  setLogs(job.logs);
  if (job.partial_text !== undefined) {
    output.value = job.partial_text || output.value;
    output.scrollTop = output.scrollHeight;
  }

  if (job.status === "succeeded") {
    return { done: true, result: job.result || {} };
  }
  if (job.status === "failed") {
    throw new Error(job.error || "Transcription failed");
  }
  return { done: false, result: null };
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (!res.ok) {
      throw new Error(job.detail || "Job status failed");
    }

    const state = applyJobSnapshot(job);
    if (state.done) {
      return state.result;
    }

    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

async function waitForJob(jobId) {
  if (!window.EventSource) {
    return pollJob(jobId);
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let fallbackStarted = false;
    const source = new EventSource(`/api/jobs/${jobId}/events`);

    function finish(callback, value) {
      if (settled) {
        return;
      }
      settled = true;
      source.close();
      callback(value);
    }

    function startFallback(error) {
      if (settled || fallbackStarted) {
        return;
      }
      fallbackStarted = true;
      source.close();
      console.warn("Job event stream failed, falling back to polling.", error);
      pollJob(jobId).then((result) => finish(resolve, result)).catch((err) => finish(reject, err));
    }

    source.addEventListener("job", (event) => {
      try {
        const job = JSON.parse(event.data);
        const state = applyJobSnapshot(job);
        if (state.done) {
          finish(resolve, state.result);
        }
      } catch (error) {
        finish(reject, error);
      }
    });

    source.addEventListener("error", (event) => {
      startFallback(event);
    });
  });
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
      summaryOutput.classList.add("empty");
      summaryOutput.textContent = "未配置 LLM_API_KEY 或 DEEPSEEK_API_KEY，配置后可提炼要点";
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
  setDownload(downloadAudio, "");
  setDownload(downloadSummary, "");
  setDownload(downloadSummaryJson, "");
  submit.disabled = true;
  submit.querySelector("span").textContent = "转写中";
  output.value = "";
  renderSummaryMarkdown("");
  jobMeta.textContent = "上传中";
  setProgress(0);
  setLogs(["准备上传文件"]);
  setStatus("模型处理中", "ready");

  try {
    const formData = new FormData(form);
    const res = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });
    let data = await res.json();
    if (!res.ok) {
      throw new Error(errorMessage(data, "Transcription failed"));
    }
    jobMeta.textContent = `${data.status} · ${data.progress}%`;
    setProgress(data.progress);
    setLogs([`任务已提交：${data.id}`]);
    data = await waitForJob(data.id);
    output.value = data.text || "";
    setDownload(downloadAudio, data.audio_url);
    setDownload(downloadText, data.text_url);
    setDownload(downloadJson, data.json_url);
    if (data.audio_duration_sec && data.input_duration_sec) {
      logs.textContent += `\n时长核对：输入 ${data.input_duration_sec}s / ASR ${data.audio_duration_sec}s`;
    }
    if (data.text_chars !== undefined) {
      logs.textContent += `\n输出字符数：${data.text_chars}；max_new_tokens=${data.max_new_tokens}`;
      logs.scrollTop = logs.scrollHeight;
    }
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
  summaryOutput.classList.add("empty");
  summaryOutput.textContent = "正在提炼要点...";
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
      throw new Error(errorMessage(data, "Summary failed"));
    }
    renderSummaryMarkdown(data.summary || "");
    setDownload(downloadSummary, data.summary_url);
    setDownload(downloadSummaryJson, data.json_url);
    setStatus(`提炼完成 · ${data.model}`, "ready");
  } catch (error) {
    summaryOutput.classList.add("empty");
    summaryOutput.textContent = error.message || String(error);
    setStatus("提炼失败", "error");
  } finally {
    summarize.disabled = false;
    summarize.textContent = "提炼要点";
  }
});

loadHealth();
