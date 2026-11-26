// app.js
// Talks to the FastAPI backend (main.py) with fetch() so the page never reloads.

// ---------- Grab the HTML elements we need ----------
const providerDropdown = document.getElementById("provider-dropdown");
const providerButton = document.getElementById("provider-button");
const providerMenu = document.getElementById("provider-menu");
const providerName = document.getElementById("provider-name");
const providerDot = document.getElementById("provider-dot");
const newChatButton = document.getElementById("new-chat-button");

const refreshButton = document.getElementById("refresh-button");
const refreshLabel = document.getElementById("refresh-label");

const uploadForm = document.getElementById("upload-form");
const uploadInput = document.getElementById("upload-input");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const dropZone = document.getElementById("drop-zone");
const fileChip = document.getElementById("file-chip");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");

const folderBox = document.getElementById("folder-box");
const folderInput = document.getElementById("folder-input");
const changeFolderButton = document.getElementById("change-folder-button");
const changeFolderLabel = document.getElementById("change-folder-label");
const openFolderButton = document.getElementById("open-folder-button");
const openFolderLabel = document.getElementById("open-folder-label");

const historyList = document.getElementById("history-list");
const notesList = document.getElementById("notes-list");
const papersList = document.getElementById("papers-list");

const selectedPaperBox = document.getElementById("selected-paper");
const selectedPaperText = document.getElementById("selected-paper-text");
const clearPaperButton = document.getElementById("clear-paper-button");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const inputWrap = document.getElementById("input-wrap");
const sendButton = document.getElementById("send-button");
const typingNote = document.getElementById("typing-note");
const inputHint = document.getElementById("input-hint");
const charCount = document.getElementById("char-count");
const chatMessages = document.getElementById("chat-messages");

// One conversation = one thread id. null means "start a new conversation".
let threadId = null;

// The paper chosen from the sidebar ({paper_id, title}). null means "no paper selected".
let selectedPaper = null;

// The provider chosen in the dropdown.
let currentProvider = "gemini";

// The saved download folder (so Cancel can put it back).
let savedFolder = "";
let editingFolder = false;

// Timers for the small animations.
let typingTimer = null;

// The backend puts this line in front of a message when a paper is selected.
const PAPER_LINE_PATTERN = /^\[Selected paper: ([^\]\s]+)[^\]]*\]\n/;

const TOOL_LABELS = {
    search_arxiv: "Searching arXiv",
    list_uploaded_papers: "Checking uploaded papers",
    load_paper: "Reading paper",
    summarize_paper: "Summarizing paper",
    ask_paper: "Asking the paper",
    compare_papers: "Comparing papers",
    save_notes: "Saving notes",
};

const PROVIDER_INFO = {
    gemini: { model: "Gemini 2.5 Flash · Google", color: "#2563eb" },
    groq: { model: "GPT-OSS 120B · Groq", color: "#b45309" },
    openrouter: { model: "Nemotron 3 Ultra (free) · OpenRouter", color: "#0f766e" },
};


// ---------- Helper: make text safe, then turn Markdown into HTML ----------
function escapeHTML(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function markdownToHTML(text) {
    return marked.parse(escapeHTML(text));
}

// Builds a line icon from the icons at the top of index.html
function makeIcon(name, size) {
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("class", "icon " + (size || ""));
    icon.setAttribute("aria-hidden", "true");

    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#i-" + name);
    icon.appendChild(use);
    return icon;
}

function makeSpan(className, text) {
    const span = document.createElement("span");
    span.className = className;
    if (text) {
        span.textContent = text;
    }
    return span;
}


// ---------- Chat helpers ----------
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessage(role, text) {
    const message = document.createElement("div");
    message.className = "message " + role;

    if (role === "assistant") {
        message.innerHTML = markdownToHTML(text);
    } else {
        message.textContent = text;
    }

    chatMessages.appendChild(message);
    scrollToBottom();
    return message;
}

function addUserMessage(text, paperId) {
    const message = document.createElement("div");
    message.className = "message user";

    if (paperId) {
        message.appendChild(makeSpan("paper-label", "About " + paperId));
    }

    message.appendChild(document.createTextNode(text));
    chatMessages.appendChild(message);
    scrollToBottom();
}

function resetChat() {
    threadId = null;
    clearSelectedPaper();
    chatMessages.innerHTML = "";
    addMessage("assistant", "New chat started. What would you like to research?");
    loadThreads();
}


// ---------- Agent steps (live tool progress) ----------
function makeStepRow(title, sub, iconName) {
    const step = document.createElement("div");
    step.className = "step";

    const badge = makeSpan("step-badge");
    badge.appendChild(makeIcon(iconName));
    step.appendChild(badge);

    const textBox = makeSpan("step-text");
    textBox.appendChild(makeSpan("step-title", title));
    textBox.appendChild(makeSpan("step-sub", sub));
    step.appendChild(textBox);
    return step;
}

function makeThinkingRow() {
    const step = makeStepRow("Thinking", "Working on your request", "loader");
    step.classList.add("thinking");

    const dots = makeSpan("dots");
    dots.setAttribute("aria-hidden", "true");
    dots.appendChild(makeSpan(""));
    dots.appendChild(makeSpan(""));
    dots.appendChild(makeSpan(""));
    step.appendChild(dots);
    return step;
}

function addStepsBox() {
    const stepsBox = document.createElement("div");
    stepsBox.className = "steps";
    stepsBox.appendChild(makeSpan("steps-label", "Agent steps"));
    stepsBox.appendChild(makeThinkingRow());

    chatMessages.appendChild(stepsBox);
    scrollToBottom();
    return stepsBox;
}

function getStepDetails(toolName, args) {
    let details = toolName;

    if (args && args.paper_id) {
        details += " · " + args.paper_id;
    }
    if (args && args.paper_ids) {
        details += " · " + args.paper_ids.join(", ");
    }
    if (args && args.query) {
        details += ' · "' + args.query + '"';
    }
    return details;
}

function removeThinking(stepsBox) {
    const thinking = stepsBox.querySelector(".step.thinking");
    if (thinking) {
        thinking.remove();
    }
}

function addStep(stepsBox, toolName, args) {
    removeThinking(stepsBox);

    const title = TOOL_LABELS[toolName] || toolName;
    const step = makeStepRow(title, getStepDetails(toolName, args), "loader");
    step.classList.add("tool");
    step.dataset.tool = toolName;

    stepsBox.appendChild(step);
    scrollToBottom();
}

function markStepDone(stepsBox, toolName) {
    const steps = stepsBox.querySelectorAll(".step.tool:not(.done)");

    for (const step of steps) {
        if (step.dataset.tool === toolName) {
            step.classList.add("done");
            step.querySelector("use").setAttribute("href", "#i-check");
            step.appendChild(makeSpan("step-state", "Done"));
            break;
        }
    }
}

function finishSteps(stepsBox) {
    removeThinking(stepsBox);

    if (stepsBox.querySelectorAll(".step").length === 0) {
        stepsBox.remove();
    }
}


// ---------- A. Provider dropdown ----------
function setProvider(provider) {
    currentProvider = provider;
    providerName.textContent = provider;

    let color = "#2563eb";
    if (PROVIDER_INFO[provider]) {
        color = PROVIDER_INFO[provider].color;
    }
    providerDot.style.background = color;

    for (const option of providerMenu.querySelectorAll(".dropdown-option")) {
        option.classList.toggle("active", option.dataset.provider === provider);
        option.parentElement.setAttribute("aria-selected", option.dataset.provider === provider);
    }
}

function openMenu() {
    providerMenu.classList.remove("hidden");
    providerDropdown.classList.add("open");
    providerButton.setAttribute("aria-expanded", "true");
}

function closeMenu() {
    providerMenu.classList.add("hidden");
    providerDropdown.classList.remove("open");
    providerButton.setAttribute("aria-expanded", "false");
}

function makeProviderOption(provider) {
    const item = document.createElement("li");
    item.setAttribute("role", "option");

    const button = document.createElement("button");
    button.type = "button";
    button.className = "dropdown-option";
    button.dataset.provider = provider;

    let info = { model: "", color: "#2563eb" };
    if (PROVIDER_INFO[provider]) {
        info = PROVIDER_INFO[provider];
    }

    const dot = makeSpan("provider-dot");
    dot.style.background = info.color;
    dot.style.width = "10px";
    dot.style.height = "10px";
    button.appendChild(dot);

    const textBox = makeSpan("option-text");
    textBox.appendChild(makeSpan("option-name", provider));
    textBox.appendChild(makeSpan("option-model", info.model));
    button.appendChild(textBox);

    const check = makeIcon("check", "small");
    check.classList.add("option-check");
    button.appendChild(check);

    button.addEventListener("click", () => {
        setProvider(provider);
        closeMenu();
        providerButton.focus();
    });

    item.appendChild(button);
    return item;
}

async function loadProviders() {
    try {
        const response = await fetch("/api/providers");
        const data = await response.json();

        providerMenu.innerHTML = "";
        for (const provider of data.providers) {
            providerMenu.appendChild(makeProviderOption(provider));
        }
        setProvider(data.default);
    } catch (error) {
        addMessage("error", "⚠ Cannot reach the server. Is main.py running?");
    }
}


// ---------- B. Load saved notes and papers into the sidebar ----------
async function loadNotes() {
    const response = await fetch("/api/notes");
    const data = await response.json();

    notesList.innerHTML = "";
    if (data.notes.length === 0) {
        notesList.innerHTML = '<li class="muted">No notes yet.</li>';
        return;
    }

    for (const note of data.notes) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = "/api/notes/" + encodeURIComponent(note.name);
        link.textContent = note.name;
        item.appendChild(link);
        item.appendChild(makeSpan("muted", " (" + note.size_kb + " KB)"));
        notesList.appendChild(item);
    }
}

function getPaperStatus(paper) {
    if (paper.uploaded) {
        return "Uploaded";
    }
    if (paper.summarized) {
        return "Summarized";
    }
    if (paper.loaded) {
        return "PDF";
    }
    return "Found";
}

async function loadPapers() {
    const response = await fetch("/api/papers");
    const data = await response.json();

    papersList.innerHTML = "";
    if (data.papers.length === 0) {
        papersList.innerHTML = '<li class="muted">No papers yet.</li>';
        return;
    }

    for (const paper of data.papers) {
        const isSelected = selectedPaper && selectedPaper.paper_id === paper.paper_id;

        const item = document.createElement("li");
        item.className = "paper-row";
        if (isSelected) {
            item.classList.add("active");
        }

        // 1. The paper itself: click it to ask about it in the chat
        const selectButton = document.createElement("button");
        selectButton.type = "button";
        selectButton.className = "paper-select";
        selectButton.title = "Ask about this paper";
        selectButton.setAttribute("aria-pressed", isSelected ? "true" : "false");

        const meta = makeSpan("paper-meta");
        meta.appendChild(makeSpan("paper-id", paper.paper_id));
        meta.appendChild(makeSpan("status-tag", getPaperStatus(paper)));
        if (isSelected) {
            const tag = makeSpan("selected-tag");
            tag.appendChild(makeIcon("check", "small"));
            tag.appendChild(document.createTextNode("Asking about this"));
            meta.appendChild(tag);
        }
        selectButton.appendChild(meta);
        selectButton.appendChild(makeSpan("paper-title", paper.title));
        selectButton.addEventListener("click", () => selectPaper(paper));
        item.appendChild(selectButton);

        // 2. A small link to the arXiv page (arXiv papers only)
        if (!paper.uploaded) {
            const link = document.createElement("a");
            link.className = "paper-link";
            link.href = "https://arxiv.org/abs/" + paper.paper_id;
            link.target = "_blank";
            link.title = "Open on arXiv";
            link.setAttribute("aria-label", "Open " + paper.paper_id + " on arXiv");
            link.appendChild(makeIcon("external", "small"));
            item.appendChild(link);
        }

        // 3. The delete button
        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "delete-button";
        deleteButton.title = "Delete this paper";
        deleteButton.setAttribute("aria-label", "Delete " + paper.paper_id);
        deleteButton.appendChild(makeIcon("trash", "small"));
        deleteButton.addEventListener("click", () => deletePaper(paper));
        item.appendChild(deleteButton);

        papersList.appendChild(item);
    }
}

async function refreshSidebar() {
    try {
        await loadNotes();
        await loadPapers();
    } catch (error) {
        console.log("Could not refresh the sidebar:", error);
    }
}

async function refreshWithAnimation() {
    refreshButton.classList.add("spinning");
    refreshLabel.textContent = "Refreshing";

    await refreshSidebar();

    setTimeout(() => {
        refreshButton.classList.remove("spinning");
        refreshLabel.textContent = "Refresh";
    }, 1200);
}


// ---------- C. Read the live stream from /api/chat/stream ----------
function handleEvent(event, stepsBox) {
    if (event.type === "start") {
        threadId = event.thread_id;
    } else if (event.type === "tool_call") {
        addStep(stepsBox, event.name, event.args);
    } else if (event.type === "tool_result") {
        markStepDone(stepsBox, event.name);
    } else if (event.type === "final") {
        finishSteps(stepsBox);
        addMessage("assistant", event.answer);
    } else if (event.type === "error") {
        finishSteps(stepsBox);
        addMessage("error", "⚠ " + event.message);
    } else if (event.type === "done") {
        finishSteps(stepsBox);
    }
}

async function streamChat(message, paperId, stepsBox) {
    const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message: message,
            provider: currentProvider,
            thread_id: threadId,
            paper_id: paperId,
        }),
    });

    if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Something went wrong.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
            if (part.startsWith("data: ")) {
                const event = JSON.parse(part.slice(6));
                handleEvent(event, stepsBox);
            }
        }
    }
}


// ---------- D. Chat form (typing animation + Send button) ----------
function updateInputState() {
    const text = chatInput.value.trim();

    sendButton.classList.toggle("ready", text.length > 0);
    if (text.length > 0) {
        charCount.textContent = chatInput.value.length + " characters";
    } else {
        charCount.textContent = "";
    }
}

function showTyping() {
    inputWrap.classList.add("typing");
    sendButton.classList.add("typing");
    typingNote.classList.remove("hidden");
    inputHint.classList.add("hidden");

    clearTimeout(typingTimer);
    typingTimer = setTimeout(hideTyping, 900);
}

function hideTyping() {
    inputWrap.classList.remove("typing");
    sendButton.classList.remove("typing");
    typingNote.classList.add("hidden");
    inputHint.classList.remove("hidden");
}

function playSendAnimation() {
    sendButton.classList.add("sending");
    setTimeout(() => sendButton.classList.remove("sending"), 650);
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();   // stop the form from reloading the page

    const message = chatInput.value.trim();
    if (!message) {
        chatInput.focus();
        return;
    }

    let paperId = null;
    if (selectedPaper) {
        paperId = selectedPaper.paper_id;
    }

    playSendAnimation();
    hideTyping();
    addUserMessage(message, paperId);
    chatInput.value = "";
    updateInputState();
    sendButton.disabled = true;
    chatInput.disabled = true;
    const stepsBox = addStepsBox();

    try {
        await streamChat(message, paperId, stepsBox);
    } catch (error) {
        finishSteps(stepsBox);
        addMessage("error", "⚠ " + error.message);
    } finally {
        sendButton.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
        refreshSidebar();
        loadThreads();
    }
});

chatInput.addEventListener("input", () => {
    updateInputState();
    showTyping();
});

chatInput.addEventListener("focus", () => inputWrap.classList.add("focused"));
chatInput.addEventListener("blur", () => inputWrap.classList.remove("focused"));

// Enter sends the message, Shift+Enter makes a new line
chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});


// ---------- E. Upload a PDF paper ----------
function setUploadStatus(text, type) {
    uploadStatus.textContent = text;
    uploadStatus.className = "upload-status " + type;
}

function showChosenFile() {
    const file = uploadInput.files[0];

    if (!file) {
        fileChip.classList.add("hidden");
        return;
    }

    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024 * 1024)).toFixed(1) + " MB";
    fileChip.classList.remove("hidden");
    setUploadStatus("", "muted");
}

uploadInput.addEventListener("change", showChosenFile);

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");

    if (event.dataTransfer.files.length > 0) {
        uploadInput.files = event.dataTransfer.files;
        showChosenFile();
    }
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = uploadInput.files[0];
    if (!file) {
        setUploadStatus("Please choose a PDF file first.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadButton.disabled = true;
    setUploadStatus("Uploading and reading " + file.name + "...", "muted");

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        setUploadStatus("Uploaded as " + data.paper_id + " (" + data.pages + " pages)", "success");
        addMessage(
            "assistant",
            "Uploaded **" + data.title + "** as `" + data.paper_id + "` (" + data.pages + " pages).\n\n" +
            "You can now summarize it, ask questions about it, or compare it with other papers."
        );
        chatInput.value = "Summarize the uploaded paper " + data.paper_id;
        updateInputState();
        chatInput.focus();
        uploadInput.value = "";
        showChosenFile();
        refreshSidebar();
    } catch (error) {
        setUploadStatus("⚠ " + error.message, "error");
    } finally {
        uploadButton.disabled = false;
    }
});


// ---------- F. Chat history (saved in SQLite by the backend) ----------
async function loadThreads() {
    try {
        const response = await fetch("/api/threads");
        const data = await response.json();

        historyList.innerHTML = "";
        if (data.threads.length === 0) {
            historyList.innerHTML = '<li class="muted">No chats yet.</li>';
            return;
        }

        for (const thread of data.threads) {
            const item = document.createElement("li");
            const button = document.createElement("button");
            button.type = "button";
            button.className = "history-item";
            button.textContent = thread.title;
            if (thread.thread_id === threadId) {
                button.classList.add("active");
            }
            button.addEventListener("click", () => openThread(thread.thread_id));

            item.appendChild(button);
            historyList.appendChild(item);
        }
    } catch (error) {
        console.log("Could not load the chat history:", error);
    }
}

async function openThread(id) {
    try {
        const response = await fetch("/api/threads/" + encodeURIComponent(id));
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not open this chat.");
        }

        threadId = data.thread_id;
        clearSelectedPaper();
        chatMessages.innerHTML = "";

        for (const message of data.messages) {
            if (message.role === "user") {
                const match = message.content.match(PAPER_LINE_PATTERN);
                if (match) {
                    addUserMessage(message.content.replace(PAPER_LINE_PATTERN, ""), match[1]);
                } else {
                    addUserMessage(message.content, null);
                }
            } else {
                addMessage("assistant", message.content);
            }
        }

        loadThreads();
    } catch (error) {
        addMessage("error", "⚠ " + error.message);
    }
}


// ---------- G. Select a paper to ask about it ----------
function selectPaper(paper) {
    selectedPaper = { paper_id: paper.paper_id, title: paper.title };

    selectedPaperText.textContent = "Asking about: " + paper.paper_id + " - " + paper.title;
    selectedPaperBox.classList.remove("hidden");

    let source = "arXiv";
    if (paper.uploaded) {
        source = "Uploaded PDF";
    }
    addMessage(
        "assistant",
        "**" + paper.title + "**\n\n" +
        "`" + paper.paper_id + "` · " + source + " · " + paper.published + "\n\n" +
        "Ask me anything about this paper, for example: *What problem does it solve?* or *Which datasets are used?*"
    );

    chatInput.focus();
    loadPapers();
}

function clearSelectedPaper() {
    selectedPaper = null;
    selectedPaperText.textContent = "";
    selectedPaperBox.classList.add("hidden");
}


// ---------- H. Delete a paper (asks for confirmation first) ----------
async function deletePaper(paper) {
    const question =
        "Delete " + paper.paper_id + " '" + paper.title + "'?\n\n" +
        "This removes the PDF, its summary and its saved index. This cannot be undone.";

    if (!confirm(question)) {
        return;
    }

    try {
        const response = await fetch("/api/papers/" + encodeURIComponent(paper.paper_id), {
            method: "DELETE",
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not delete the paper.");
        }

        if (selectedPaper && selectedPaper.paper_id === paper.paper_id) {
            clearSelectedPaper();
        }
        addMessage("assistant", "Deleted `" + data.deleted + "`.");
    } catch (error) {
        addMessage("error", "⚠ " + error.message);
    } finally {
        refreshSidebar();
    }
}


// ---------- I. Download folder ----------
async function loadSettings() {
    try {
        const response = await fetch("/api/settings");
        const data = await response.json();

        savedFolder = data.download_folder;
        folderInput.value = savedFolder;
        folderInput.title = savedFolder;
    } catch (error) {
        console.log("Could not load the settings:", error);
    }
}

function startEditingFolder() {
    editingFolder = true;
    folderInput.readOnly = false;
    folderBox.classList.add("editing");
    changeFolderLabel.textContent = "Save";
    openFolderLabel.textContent = "Cancel";
    folderInput.focus();
    folderInput.select();
}

function stopEditingFolder() {
    editingFolder = false;
    folderInput.readOnly = true;
    folderBox.classList.remove("editing");
    changeFolderLabel.textContent = "Change folder";
    openFolderLabel.textContent = "Open";
    folderInput.value = savedFolder;
}

async function saveFolder() {
    const path = folderInput.value.trim();
    if (!path) {
        folderInput.focus();
        return;
    }

    try {
        const response = await fetch("/api/settings/download-folder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: path }),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not change the folder.");
        }

        savedFolder = data.download_folder;
        stopEditingFolder();
        addMessage("assistant", "New papers will now be saved in `" + savedFolder + "`.");
        refreshSidebar();
    } catch (error) {
        addMessage("error", "⚠ " + error.message);
    }
}

async function openFolder() {
    try {
        const response = await fetch("/api/settings/open-folder", { method: "POST" });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not open the folder.");
        }
    } catch (error) {
        addMessage("error", "⚠ " + error.message);
    }
}

changeFolderButton.addEventListener("click", () => {
    if (editingFolder) {
        saveFolder();
    } else {
        startEditingFolder();
    }
});

openFolderButton.addEventListener("click", () => {
    if (editingFolder) {
        stopEditingFolder();
    } else {
        openFolder();
    }
});

folderInput.addEventListener("keydown", (event) => {
    if (!editingFolder) {
        return;
    }
    if (event.key === "Enter") {
        saveFolder();
    } else if (event.key === "Escape") {
        stopEditingFolder();
    }
});


// ---------- J. Buttons ----------
newChatButton.addEventListener("click", resetChat);
refreshButton.addEventListener("click", refreshWithAnimation);
clearPaperButton.addEventListener("click", () => {
    clearSelectedPaper();
    loadPapers();
});

providerButton.addEventListener("click", () => {
    if (providerDropdown.classList.contains("open")) {
        closeMenu();
    } else {
        openMenu();
    }
});

// Close the provider menu when clicking somewhere else or pressing Escape
document.addEventListener("click", (event) => {
    if (!providerDropdown.contains(event.target)) {
        closeMenu();
    }
});

providerDropdown.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        closeMenu();
        providerButton.focus();
    }
});


// ---------- K. Start ----------
loadProviders();
loadSettings();
refreshSidebar();
loadThreads();
