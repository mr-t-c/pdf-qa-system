const API = 'http://localhost:8000';

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

let selectedFile = null;          // File object chosen by user
let documents    = [];            // Array of { doc_id, filename, num_chunks }

document.addEventListener('DOMContentLoaded', () => {
  fetchDocuments();
  bindEvents();
});

function bindEvents() {

  browseBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setSelectedFile(fileInput.files[0]);
  });

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

  clearFileBtn.addEventListener('click', clearSelectedFile);

  uploadBtn.addEventListener('click', uploadPDF);

  refreshBtn.addEventListener('click', fetchDocuments);

  docSelect.addEventListener('change', updateScopeLabel);

  questionInput.addEventListener('input', () => {
    askBtn.disabled = questionInput.value.trim() === '';
  });

  askBtn.addEventListener('click', askQuestion);

  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      if (!askBtn.disabled) askQuestion();
    }
  });

  dismissError.addEventListener('click', hideError);
}


function setSelectedFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  dropInner.hidden  = true;
  fileSelected.hidden = false;
  uploadBtn.disabled  = false;
  hideError();
}

function clearSelectedFile() {
  selectedFile = null;
  fileInput.value = '';
  dropInner.hidden  = false;
  fileSelected.hidden = true;
  uploadBtn.disabled  = true;
  hideUploadMsg();
}

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

    showUploadMsg(`✓ "${data.filename}" uploaded — ${data.num_chunks} chunks indexed.`, false);

    await fetchDocuments();

    docSelect.value = data.doc_id;
    updateScopeLabel();

    clearSelectedFile();

  } catch (err) {
    showUploadMsg(err.message, true);
  } finally {
    setButtonLoading(uploadBtn, false);
  }
}

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

function renderDocuments(docs) {
  const currentVal = docSelect.value;
  docSelect.innerHTML = '<option value="">— Search all documents —</option>';
  docs.forEach(doc => {
    const opt = document.createElement('option');
    opt.value       = doc.doc_id;
    opt.textContent = doc.filename;
    docSelect.appendChild(opt);
  });

  if (docs.find(d => d.doc_id === currentVal)) {
    docSelect.value = currentVal;
  }
  updateScopeLabel();

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

async function deleteDocument(docId, filename) {
  if (!confirm(`Delete "${filename}" from the index?`)) return;

  try {
    const res = await fetch(`${API}/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || 'Delete failed.');
    }

    if (docSelect.value === docId) {
      docSelect.value = '';
      updateScopeLabel();
    }

    await fetchDocuments();

  } catch (err) {
    showError('Delete failed: ' + err.message);
  }
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  const docId = docSelect.value || null;

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

function renderAnswer(data) {
  answerSection.hidden = false;

  answerBody.textContent = data.answer;

  const pct = Math.round((data.confidence || 0) * 100);
  confidencePct.textContent = `${pct}%`;
  confidenceBar.style.width = `${pct}%`;

  if (pct >= 60) {
    confidenceBar.style.backgroundColor = 'var(--conf-high)';
  } else if (pct >= 35) {
    confidenceBar.style.backgroundColor = 'var(--conf-mid)';
  } else {
    confidenceBar.style.backgroundColor = 'var(--conf-low)';
  }

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

function clearAnswer() {
  answerSection.hidden  = true;
  answerBody.textContent = '';
  sourcesTags.innerHTML  = '';
  sourcesWrap.hidden     = true;
  confidenceBar.style.width = '0%';
  confidencePct.textContent = '0%';
}

function updateScopeLabel() {
  const selected = docSelect.options[docSelect.selectedIndex];
  if (docSelect.value) {
    scopeLabel.textContent = `Searching: ${selected.textContent}`;
  } else {
    scopeLabel.textContent = 'Searching all documents';
  }
}

function setButtonLoading(btn, loading) {
  const text    = btn.querySelector('.btn-text');
  const spinner = btn.querySelector('.spinner');
  btn.disabled  = loading;
  if (text)    text.hidden    = loading;
  if (spinner) spinner.hidden = !loading;
}

function showUploadMsg(msg, isError = false) {
  uploadMsg.textContent = msg;
  uploadMsg.className   = isError ? 'status-msg error' : 'status-msg';
  uploadMsg.hidden      = false;
}

function hideUploadMsg() {
  uploadMsg.hidden = true;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden   = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorMsg.textContent = '';
}



const topicsSection = document.getElementById('topics-section');
const topicsChips   = document.getElementById('topics-chips');
const topicsToggle  = document.getElementById('topics-toggle');
const topicsChevron = document.getElementById('topics-chevron');

let topicsExpanded = false;

topicsToggle.addEventListener('click', () => {
  topicsExpanded = !topicsExpanded;
  topicsChips.hidden = !topicsExpanded;
  topicsToggle.classList.toggle('open', topicsExpanded);
});

async function fetchTopics(docId) {
  topicsSection.hidden = false;
  topicsExpanded = false;
  topicsChips.hidden = true;
  topicsToggle.classList.remove('open');
  topicsChips.innerHTML = '<span class="topics-loading">Extracting topics...</span>';

  try {
    const res  = await fetch(`${API}/topics/${docId}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Failed to fetch topics.');

    renderTopics(data.topics);

  } catch (err) {
    topicsChips.innerHTML = '<span class="topics-loading">Could not load topics.</span>';
  }
}

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
      const question = topic.endsWith('?') ? topic : `${topic}?`;
      questionInput.value = question;

      askBtn.disabled = false;
      questionInput.focus();

      questionInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    topicsChips.appendChild(chip);
  });
}

function clearTopics() {
  topicsSection.hidden = true;
  topicsChips.innerHTML = '';
  topicsExpanded = false;
  topicsToggle.classList.remove('open');
}

const _originalUpdateScopeLabel = updateScopeLabel;
updateScopeLabel = function () {
  _originalUpdateScopeLabel();

  if (docSelect.value) {
    fetchTopics(docSelect.value);
  } else {
    clearTopics();
  }
};