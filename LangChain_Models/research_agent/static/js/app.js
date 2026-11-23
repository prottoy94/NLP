// app.js
// Talks to the FastAPI backend (main.py) with fetch() so the page never reloads.

// ---------- Grab the HTML elements we need ----------
const providerSelect = document.getElementById("provider-select");
const newChatButton = document.getElementById("new-chat-button");
const refreshButton = document.getElementById("refresh-button");

const notesList = document.getElementById("notes-list");
const papersList = document.getElementById("papers-list");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const chatMessages = document.getElementById("chat-messages");

// One conversation = one thread id. null means "start a new conversation".
let threadId = null;

const TOOL_LABELS = {
    search_arxiv: "🔍 Searching arXiv",
    load_paper: "📄 Reading paper",
    summarize_paper: "📝 Summarizing paper",
    ask_paper: "❓ Asking the paper",
    compare_papers: "⚖️ Comparing papers",
    save_notes: "💾 Saving notes",
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

function addStepsBox() {
    const stepsBox = document.createElement("div");
    stepsBox.className = "steps";
    stepsBox.innerHTML = '<div class="step"><span class="spinner"></span>Thinking...</div>';
    chatMessages.appendChild(stepsBox);
    scrollToBottom();
    return stepsBox;
}

function getToolLabel(toolName, args) {
    let label = TOOL_LABELS[toolName] || "🔧 " + toolName;

    if (args && args.arxiv_id) {
        label += " " + args.arxiv_id;
    }
    if (args && args.query) {
        label += ': "' + args.query + '"';
    }
    return label;
}

function addStep(stepsBox, toolName, args) {
    const thinking = stepsBox.querySelector(".step:not(.tool)");
    if (thinking) {
        thinking.remove();
    }

    const step = document.createElement("div");
    step.className = "step tool";
    step.dataset.tool = toolName;
    step.innerHTML = '<span class="spinner"></span>';
    step.appendChild(document.createTextNode(getToolLabel(toolName, args)));

    stepsBox.appendChild(step);
    scrollToBottom();
}

function markStepDone(stepsBox, toolName) {
    const steps = stepsBox.querySelectorAll('.step.tool:not(.done)');

    for (const step of steps) {
        if (step.dataset.tool === toolName) {
            step.classList.add("done");
            step.querySelector(".spinner").replaceWith("✅ ");
            break;
        }
    }
}

function finishSteps(stepsBox) {
    const thinking = stepsBox.querySelector(".step:not(.tool)");
    if (thinking) {
        thinking.remove();
    }

    if (stepsBox.children.length === 0) {
        stepsBox.remove();
    }
}

function resetChat() {
    threadId = null;
    chatMessages.innerHTML = "";
    addMessage("assistant", "New chat started. What would you like to research?");
}


// ---------- A. Load providers into the dropdown ----------
async function loadProviders() {
    try {
        const response = await fetch("/api/providers");
        const data = await response.json();

        providerSelect.innerHTML = "";
        for (const provider of data.providers) {
            const option = document.createElement("option");
            option.value = provider;
            option.textContent = provider;
            if (provider === data.default) {
                option.selected = true;
            }
            providerSelect.appendChild(option);
        }
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
        link.textContent = "⬇ " + note.name;
        item.appendChild(link);
        item.appendChild(document.createTextNode(" (" + note.size_kb + " KB)"));
        notesList.appendChild(item);
    }
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
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = "https://arxiv.org/abs/" + paper.arxiv_id;
        link.target = "_blank";
        link.textContent = paper.arxiv_id;
        item.appendChild(link);

        let status = "";
        if (paper.summarized) {
            status = " 📝";
        } else if (paper.loaded) {
            status = " 📄";
        }
        item.appendChild(document.createTextNode(status + " " + paper.title));
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

async function streamChat(message, stepsBox) {
    const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message: message,
            provider: providerSelect.value,
            thread_id: threadId,
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


// ---------- D. Chat form ----------
chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();   // stop the form from reloading the page

    const message = chatInput.value.trim();
    if (!message) {
        chatInput.focus();
        return;
    }

    addMessage("user", message);
    chatInput.value = "";
    sendButton.disabled = true;
    chatInput.disabled = true;
    const stepsBox = addStepsBox();

    try {
        await streamChat(message, stepsBox);
    } catch (error) {
        finishSteps(stepsBox);
        addMessage("error", "⚠ " + error.message);
    } finally {
        sendButton.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
        refreshSidebar();
    }
});

// Enter sends the message, Shift+Enter makes a new line
chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});


// ---------- E. Buttons ----------
newChatButton.addEventListener("click", resetChat);
refreshButton.addEventListener("click", refreshSidebar);


// ---------- F. Start ----------
loadProviders();
refreshSidebar();
