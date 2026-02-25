/* ============================================================
   DocMind — PDF QA Assistant
   script.js: All API calls, DOM updates, and UI state
   ============================================================ */

const API = 'http://localhost:8000';

/* ── DOM References ────────────────────────────────────────── */
const dropZone       = document.getElementById('drop-zone');
const dropInner      = document.getElementById('drop-inner');
const fileInput      = document.getElementById('file-input');
const browseBtn      = document.getElementById('browse-btn');
const fileSelected   = document.getElementById('file-selected');
const fileNameEl     = document.getElementById('file-name');
const clearFileBtn   = document.getElementById('clear-file-btn');
const uploadBtn      = document.getElementById('upload-btn');
const uploadMsg      = document.getElementById('upload-msg');

const refreshBtn     = document.getElementById('refresh-btn');
const docSelect      = document.getElementById('doc-select');
const docList        = document.getElementById('doc-list');

const questionInput  = document.getElementById('question-input');
const scopeLabel     = document.getElementById('scope-label');
const askBtn         = document.getElementById('ask-btn');

const answerSection  = document.getElementById('answer-section');
const answerBody     = document.getElementById('answer-body');
const confidenceBar  = document.getElementById('confidence-bar');
const confidencePct  = document.getElementById('confidence-pct');
const sourcesWrap    = document.getElementById('sources-wrap');
const sourcesTags    = document.getElementById('sources-tags');

const errorBanner    = document.getElementById('error-banner');
const errorMsg       = document.getElementById('error-msg');
const dismissError   = document.getElementById('dismiss-error');

/* ── State ─────────────────────────────────────────────────── */
let selectedFile = null;          // File object chosen by user
let documents    = [];            // Array of { doc_id, filename, num_chunks }

/* ── Initialisation ────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  fetchDocuments();
  bindEvents();
});

/* ── Event Bindings ────────────────────────────────────────── */
function bindEvents() {

  // Browse button opens file picker
  browseBtn.addEventListener('click', () => fileInput.click());

  // File picker change
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setSelectedFile(fileInput.files[0]);
  });

  // Drag-and-drop on drop zone
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.pdf')) {
      setSelectedFile(file);
    } else {
      showError('Only PDF files are supported.');
    }
  });

  // Clear selected file
  clearFileBtn.addEventListener('click', clearSelectedFile);

  // Upload button
  uploadBtn.addEventListener('click', uploadPDF);

  // Refresh documents list
  refreshBtn.addEventListener('click', fetchDocuments);

  // Document selector → update scope label
  docSelect.addEventListener('change', updateScopeLabel);

  // Question input → enable Ask when non-empty
  questionInput.addEventListener('input', () => {
    askBtn.disabled = questionInput.value.trim() === '';
  });

  // Ask button
  askBtn.addEventListener('click', askQuestion);

  // Allow Ctrl+Enter to submit question
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      if (!askBtn.disabled) askQuestion();
    }
  });

  // Dismiss error banner
  dismissError.addEventListener('click', hideError);
}

/* ── File Selection Helpers ────────────────────────────────── */

/**
 * Store the chosen file and update the drop zone UI to show the file name.
 */
function setSelectedFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  dropInner.hidden  = true;
  fileSelected.hidden = false;
  uploadBtn.disabled  = false;
  hideError();
}

/**
 * Clear the chosen file and reset the drop zone to its default state.
 */
function clearSelectedFile() {
  selectedFile = null;
  fileInput.value = '';
  dropInner.hidden  = false;
  fileSelected.hidden = true;
  uploadBtn.disabled  = true;
  hideUploadMsg();
}

/* ── Upload ─────────────────────────────────────────────────── */

/**
 * POST /upload with the selected PDF as multipart/form-data.
 * On success, refresh the document list and show confirmation.
 */
async function uploadPDF() {
  if (!selectedFile) return;

  setButtonLoading(uploadBtn, true);
  hideUploadMsg();
  hideError();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res  = await fetch(`${API}/upload`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Upload failed.');

    // Show success message
    showUploadMsg(`✓ "${data.filename}" uploaded — ${data.num_chunks} chunks indexed.`, false);

    // Refresh document list so new doc appears immediately
    await fetchDocuments();

    // Auto-select the just-uploaded document
    docSelect.value = data.doc_id;
    updateScopeLabel();

    clearSelectedFile();

  } catch (err) {
    showUploadMsg(err.message, true);
  } finally {
    setButtonLoading(uploadBtn, false);
  }
}

/* ── Documents ──────────────────────────────────────────────── */

/**
 * GET /documents and rebuild both the dropdown and the document list panel.
 */
async function fetchDocuments() {
  try {
    const res  = await fetch(`${API}/documents`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to fetch documents.');

    documents = data;
    renderDocuments(data);

  } catch (err) {
    showError('Could not load documents: ' + err.message);
  }
}

/**
 * Rebuild the <select> dropdown and the sidebar document list from the
 * current documents array.
 */
function renderDocuments(docs) {
  // Rebuild dropdown — keep the "all" placeholder option first
  const currentVal = docSelect.value;
  docSelect.innerHTML = '<option value="">— Search all documents —</option>';
  docs.forEach(doc => {
    const opt = document.createElement('option');
    opt.value       = doc.doc_id;
    opt.textContent = doc.filename;
    docSelect.appendChild(opt);
  });

  // Restore selection if it still exists
  if (docs.find(d => d.doc_id === currentVal)) {
    docSelect.value = currentVal;
  }
  updateScopeLabel();

  // Rebuild sidebar list
  docList.innerHTML = '';
  if (docs.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'doc-list-empty';
    empty.textContent = 'No documents uploaded yet.';
    docList.appendChild(empty);
    return;
  }

  docs.forEach(doc => {
    const li = document.createElement('li');
    li.className = 'doc-item';

    const nameSpan = document.createElement('span');
    nameSpan.className   = 'doc-item-name';
    nameSpan.textContent = doc.filename;
    nameSpan.title       = doc.filename;

    const chunksSpan = document.createElement('span');
    chunksSpan.className   = 'doc-item-chunks';
    chunksSpan.textContent = `${doc.num_chunks}c`;  // e.g. "42c" = 42 chunks
    chunksSpan.title       = `${doc.num_chunks} chunks`;

    const delBtn = document.createElement('button');
    delBtn.className   = 'doc-delete-btn';
    delBtn.textContent = '✕';
    delBtn.title       = 'Delete document';
    delBtn.addEventListener('click', () => deleteDocument(doc.doc_id, doc.filename));

    li.appendChild(nameSpan);
    li.appendChild(chunksSpan);
    li.appendChild(delBtn);
    docList.appendChild(li);
  });
}

/**
 * DELETE /documents/{doc_id} — remove document from index and refresh list.
 */
async function deleteDocument(docId, filename) {
  if (!confirm(`Delete "${filename}" from the index?`)) return;

  try {
    const res = await fetch(`${API}/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || 'Delete failed.');
    }

    // If the deleted doc was selected, reset to "all"
    if (docSelect.value === docId) {
      docSelect.value = '';
      updateScopeLabel();
    }

    await fetchDocuments();

  } catch (err) {
    showError('Delete failed: ' + err.message);
  }
}

/* ── Question / Answer ──────────────────────────────────────── */

/**
 * POST /ask with the typed question and selected doc_id.
 * Renders the answer, sources, and confidence score on success.
 */
async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  const docId = docSelect.value || null;

  // Clear previous answer and show loading state
  clearAnswer();
  setButtonLoading(askBtn, true);
  hideError();

  const body = { question };
  if (docId) body.doc_id = docId;

  try {
    const res  = await fetch(`${API}/ask`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Request failed.');

    renderAnswer(data);

  } catch (err) {
    showError('Could not get an answer: ' + err.message);
  } finally {
    setButtonLoading(askBtn, false);
  }
}

/**
 * Populate the answer card with the API response.
 * data = { answer, sources, confidence, doc_id, question }
 */
function renderAnswer(data) {
  // Show the answer section
  answerSection.hidden = false;

  // Answer text
  answerBody.textContent = data.answer;

  // Confidence bar and percentage
  const pct = Math.round((data.confidence || 0) * 100);
  confidencePct.textContent = `${pct}%`;
  confidenceBar.style.width = `${pct}%`;

  // Color the bar based on confidence level
  if (pct >= 60) {
    confidenceBar.style.backgroundColor = 'var(--conf-high)';
  } else if (pct >= 35) {
    confidenceBar.style.backgroundColor = 'var(--conf-mid)';
  } else {
    confidenceBar.style.backgroundColor = 'var(--conf-low)';
  }

  // Sources tags
  sourcesTags.innerHTML = '';
  if (data.sources && data.sources.length > 0) {
    sourcesWrap.hidden = false;
    data.sources.forEach(src => {
      const tag = document.createElement('span');
      tag.className   = 'source-tag';
      tag.textContent = src;
      sourcesTags.appendChild(tag);
    });
  } else {
    sourcesWrap.hidden = true;
  }
}

/**
 * Hide and reset the answer section before a new request.
 */
function clearAnswer() {
  answerSection.hidden  = true;
  answerBody.textContent = '';
  sourcesTags.innerHTML  = '';
  sourcesWrap.hidden     = true;
  confidenceBar.style.width = '0%';
  confidencePct.textContent = '0%';
}

/* ── UI Helpers ─────────────────────────────────────────────── */

/**
 * Update the scope label below the question input to reflect which
 * document (or all) the next question will search.
 */
function updateScopeLabel() {
  const selected = docSelect.options[docSelect.selectedIndex];
  if (docSelect.value) {
    scopeLabel.textContent = `Searching: ${selected.textContent}`;
  } else {
    scopeLabel.textContent = 'Searching all documents';
  }
}

/**
 * Toggle loading state on a button — shows/hides spinner and text.
 */
function setButtonLoading(btn, loading) {
  const text    = btn.querySelector('.btn-text');
  const spinner = btn.querySelector('.spinner');
  btn.disabled  = loading;
  if (text)    text.hidden    = loading;
  if (spinner) spinner.hidden = !loading;
}

/**
 * Show a status message below the upload button.
 * isError controls whether it uses error styling.
 */
function showUploadMsg(msg, isError = false) {
  uploadMsg.textContent = msg;
  uploadMsg.className   = isError ? 'status-msg error' : 'status-msg';
  uploadMsg.hidden      = false;
}

function hideUploadMsg() {
  uploadMsg.hidden = true;
}

/**
 * Show the global error banner at the bottom of the right column.
 */
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden   = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorMsg.textContent = '';
}


/* ── Topics ─────────────────────────────────────────────────── */

const topicsSection = document.getElementById('topics-section');
const topicsChips   = document.getElementById('topics-chips');

/**
 * Fetch topics for the selected document and render as clickable chips.
 * Called whenever the document selector changes to a specific document.
 */
async function fetchTopics(docId) {
  // Show section in loading state
  topicsSection.hidden = false;
  topicsChips.innerHTML = '<span class="topics-loading">Extracting topics...</span>';

  try {
    const res  = await fetch(`${API}/topics/${docId}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Failed to fetch topics.');

    renderTopics(data.topics);

  } catch (err) {
    topicsChips.innerHTML = `<span class="topics-loading">Could not load topics.</span>`;
  }
}

/**
 * Render topic strings as clickable chip buttons.
 * Clicking a chip pre-fills the question input and focuses it.
 */
function renderTopics(topics) {
  topicsChips.innerHTML = '';

  if (!topics || topics.length === 0) {
    topicsChips.innerHTML = '<span class="topics-loading">No topics found in this document.</span>';
    return;
  }

  topics.forEach(topic => {
    const chip = document.createElement('button');
    chip.className   = 'topic-chip';
    chip.textContent = topic;

    chip.addEventListener('click', () => {
      // Pre-fill the question input with the topic as a question
      const question = topic.endsWith('?') ? topic : `${topic}?`;
      questionInput.value = question;

      // Enable the Ask button and focus the input
      askBtn.disabled = false;
      questionInput.focus();

      // Scroll the question section into view on mobile
      questionInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    topicsChips.appendChild(chip);
  });
}

/**
 * Hide and clear the topics section (shown when "all documents" is selected).
 */
function clearTopics() {
  topicsSection.hidden  = true;
  topicsChips.innerHTML = '';
}

// ── Hook into document selector change ───────────────────────

// Override the existing updateScopeLabel to also handle topics
const _originalUpdateScopeLabel = updateScopeLabel;
updateScopeLabel = function () {
  _originalUpdateScopeLabel();

  if (docSelect.value) {
    // A specific document is selected — fetch its topics
    fetchTopics(docSelect.value);
  } else {
    // "All documents" — hide topics section
    clearTopics();
  }
};