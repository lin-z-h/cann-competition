const crypto = require('crypto');
const fs = require('fs');

const BASE_URL = 'https://cannjudge.cn';
const USER_ID = '6aabe4f6b0477ec41ec44882';
const PROBLEM_ID = '6a9aa054bf41025d6014f3ef';
const action = process.argv[2] || 'check';

function loadCookie() {
  const state = JSON.parse(fs.readFileSync('.tmp_cannjudge_state.json', 'utf8'));
  return state.cookies
    .filter(item => item.domain === 'cannjudge.cn' || item.domain === '.cannjudge.cn')
    .map(item => `${item.name}=${item.value}`)
    .join('; ');
}

const cookie = loadCookie();

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      accept: 'application/json',
      cookie,
      origin: BASE_URL,
      referer: `${BASE_URL}/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit`,
      ...options.headers,
    },
  });
  const text = await response.text();
  let body;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { text: text.slice(0, 300) };
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${JSON.stringify(body)}`);
  }
  return body;
}

async function getLatest() {
  const list = await request(`/api/submissions/user/${USER_ID}/problem/${PROBLEM_ID}`);
  if (!Array.isArray(list) || !list.length) throw new Error('No submissions found');
  const summary = list[0];
  const detail = await request(`/api/submissions/${summary._id}`);
  return { summary, detail };
}

function parseFiles(detail) {
  const raw = detail.files;
  const files = typeof raw === 'string' ? JSON.parse(raw) : raw;
  if (!Array.isArray(files)) throw new Error('Latest submission has no reusable files array');
  return files;
}

function compact(latest) {
  const files = parseFiles(latest.detail);
  const kernel = files.find(file => file.path === 'kernel.asc');
  const kernelText = kernel ? String(kernel.content).replace(/\r\n/g, '\n') : '';
  const results = Array.isArray(latest.detail.result) ? latest.detail.result : [];
  return {
    ID: latest.summary.ID,
    id: latest.summary._id,
    status: latest.detail.status || latest.summary.status,
    canViewCode: latest.detail.can_view_code,
    cases: results.length,
    states: [...new Set(results.map(item => item.testcase_status))],
    times: results.map(item => item.time),
    bestTimes: results.map(item => item.best_time),
    precision: results.map(item => item.precision_ratio),
    resultFields: results[0] ? Object.keys(results[0]) : [],
    testcaseIds: results.map(item => item.testcase_id),
    kernelBytes: Buffer.byteLength(kernelText),
    kernelSha256: kernelText
      ? crypto.createHash('sha256').update(kernelText).digest('hex')
      : null,
    markers: {
      probeMode: kernelText.includes('BMMS_PROBE_MODE'),
      tailStride: kernelText.includes('BMMS_TAIL_STRIDE'),
      bf16CrossLayoutGroup: kernelText.includes(
        'info_x1.tensors[0].dtype == kJudgeBfloat16'),
      asyncIterate: kernelText.includes('Iterate<false>'),
    },
    files: files.map(file => ({ path: file.path, editable: file.editable })),
  };
}

async function main() {
  const before = await getLatest();
  if (action === 'check') {
    process.stdout.write(`${JSON.stringify(compact(before), null, 2)}\n`);
    return;
  }
  if (action !== 'submit') throw new Error(`Unknown action: ${action}`);

  const beforeStatus = String(before.detail.status || before.summary.status || '').toLowerCase();
  if (beforeStatus === 'running' || beforeStatus === 'pending') {
    throw new Error(`Latest submission ${before.summary.ID} is still ${beforeStatus}`);
  }

  const localKernel = fs.readFileSync('kernel.asc', 'utf8').replace(/\r\n/g, '\n');
  const files = parseFiles(before.detail);
  const kernel = files.find(file => file.path === 'kernel.asc');
  if (!kernel) throw new Error('kernel.asc missing from latest submission template');
  kernel.content = localKernel;

  // The submit API accepts only user-editable root .asc/.h files here. The
  // immutable problem template is injected by the judge and must not be sent.
  const editableFiles = [{ path: 'kernel.asc', content: kernel.content }];

  const payload = {
    problemId: PROBLEM_ID,
    userId: USER_ID,
    files: editableFiles,
    tiling_h: '',
    tiling_key_h: '',
    host_cpp: '',
    kernel_cpp: '',
  };
  const response = await request('/api/submissions/submit', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const after = await getLatest();
  if (String(after.summary.ID) === String(before.summary.ID)) {
    throw new Error(`Submit response did not create a new submission: ${JSON.stringify(response)}`);
  }
  const hash = crypto.createHash('sha256').update(localKernel).digest('hex');
  process.stdout.write(`${JSON.stringify({
    before: before.summary.ID,
    after: after.summary.ID,
    id: after.summary._id,
    status: after.detail.status || after.summary.status,
    kernelBytes: Buffer.byteLength(localKernel),
    kernelSha256: hash,
  }, null, 2)}\n`);
}

main().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
