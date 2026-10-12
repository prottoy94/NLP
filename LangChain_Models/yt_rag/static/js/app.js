// app.js
// Talks to the Flask backend with fetch() so the page never reloads.

// ---------- Grab the HTML elements we need ----------
const loadForm = document.getElementById("load-form");
const urlInput = document.getElementById("url-input");
const loadButton = document.getElementById("load-button");
const loadStatus = document.getElementById("load-status");

const videoArea = document.getElementById("video-area");
const videoFrame = document.getElementById("video-frame");
const videoTitle = document.getElementById("video-title");
const videoIdText = document.getElementById("video-id");
const chunkCount = document.getElementById("chunk-count");

const summaryButton = document.getElementById("summary-button");
const summaryStatus = document.getElementById("summary-status");
const summaryOutput = document.getElementById("summary-output");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const chatMessages = document.getElementById("chat-messages");


// ---------- Helper: send JSON to the backend and get JSON back ----------
async function postJSON(url, body) {
    let response;
    try {
        response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body || {}),
        });
    } catch (networkError) {
        // The server is not running or the connection dropped
        throw new Error("Cannot reach the server. Is app.py still running?");
    }

    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error("The server sent an unexpected response.");
    }

    if (!data.success) {
        throw new Error(data.error || "Something went wrong.");
    }
    return data;
}


// ---------- Helper: show a status box (loading / success / error) ----------
function showStatus(element, type, message) {
    element.className = "status " + type;
    element.innerHTML = "";
    if (type === "loading") {
        const spinner = document.createElement("div");
        spinner.className = "spinner";
        element.appendChild(spinner);
    }
    const text = document.createElement("span");
    text.textContent = message;   // textContent = safe, no HTML injection
    element.appendChild(text);
}

function hideStatus(element) {
    element.className = "status hidden";
}


// ---------- Helper: turn Gemini's simple markdown into safe HTML ----------
// Gemini often answers with **bold** and "* bullet" lines.
// We escape HTML first (for safety) and then add a few basic formats.
function escapeHTML(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function formatText(text) {
    const lines = escapeHTML(text).split("\n");
    let html = "";
    let inList = false;

    for (let line of lines) {
        line = line.trim()
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")   // **bold**
            .replace(/^#{1,6}\s*(.+)$/, "<strong>$1</strong>"); // # Heading

        const bullet = line.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
        if (bullet) {
            if (!inList) { html += "<ul>"; inList = true; }
            html += "<li>" + bullet[1] + "</li>";
        } else {
            if (inList) { html += "</ul>"; inList = false; }
            if (line) html += "<p>" + line + "</p>";
        }
    }
    if (inList) html += "</ul>";
    return html;
}


// ---------- Chat helpers ----------
function addMessage(role, text) {
    const message = document.createElement("div");
    message.className = "message " + role;
    if (role === "assistant") {
        message.innerHTML = formatText(text);
    } else {
        message.textContent = text;
    }
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return message;
}

function addTypingIndicator() {
    const message = document.createElement("div");
    message.className = "message assistant typing";
    message.innerHTML = '<div class="spinner"></div><span>Thinking...</span>';
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return message;
}

function resetChat() {
    chatMessages.innerHTML = "";
    addMessage("assistant", "Hi! Ask me anything about this video. I'll answer only from its transcript.");
}


// ---------- A. Load Video ----------
loadForm.addEventListener("submit", async (event) => {
    event.preventDefault();   // stop the form from reloading the page

    const url = urlInput.value.trim();
    if (!url) {
        showStatus(loadStatus, "error", "Please paste a YouTube video URL.");
        return;
    }

    loadButton.disabled = true;
    showStatus(loadStatus, "loading",
        "Fetching transcript and building the search index... this can take a little while.");

    try {
        const data = await postJSON("/api/load-video", { url: url });

        // B. Show the video information
        videoFrame.src = data.embed_url;
        videoTitle.textContent = data.title || "Video";
        videoIdText.textContent = data.video_id;
        chunkCount.textContent = data.num_chunks + " transcript chunks indexed";
        videoArea.classList.remove("hidden");

        // New video -> clear old summary and chat
        summaryOutput.innerHTML = '<p class="muted">Click "Generate Summary" to summarize the transcript.</p>';
        hideStatus(summaryStatus);
        resetChat();

        showStatus(loadStatus, "success", "Video loaded! You can now generate a summary or ask questions.");
    } catch (error) {
        showStatus(loadStatus, "error", error.message);
    } finally {
        loadButton.disabled = false;
    }
});


// ---------- C. Generate Summary ----------
summaryButton.addEventListener("click", async () => {
    summaryButton.disabled = true;
    showStatus(summaryStatus, "loading", "Generating summary from the transcript...");

    try {
        const data = await postJSON("/api/summarize");
        summaryOutput.innerHTML = formatText(data.summary);
        hideStatus(summaryStatus);
    } catch (error) {
        showStatus(summaryStatus, "error", error.message);
    } finally {
        summaryButton.disabled = false;
    }
});


// ---------- D. Chat ----------
chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = chatInput.value.trim();
    if (!question) {
        chatInput.focus();
        return;   // ignore empty questions
    }

    addMessage("user", question);
    chatInput.value = "";
    sendButton.disabled = true;
    chatInput.disabled = true;
    const typing = addTypingIndicator();

    try {
        const data = await postJSON("/api/chat", { question: question });
        typing.remove();
        addMessage("assistant", data.answer);
    } catch (error) {
        typing.remove();
        addMessage("error", "⚠ " + error.message);
    } finally {
        sendButton.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
});
