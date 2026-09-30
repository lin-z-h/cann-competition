const crypto = require('crypto');
const {
  parseArgs, selectedSource, cookieFromState, userIdFromState,
} = require('./lib/cli');

const BASE_URL = 'https://cannjudge.cn';
const options = parseArgs(process.argv.slice(2), 'check', ['check', 'preview', 'submit']);
let cookie;
let userId;

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
  const list = await request(
    `/api/submissions/user/${userId}/problem/${options.problemId}`);
  if (!Array.isArray(list)) throw new Error('提交列表格式无效');
  if (!list.length) return null;
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
  if (options.help) {
    process.stdout.write('用法：node tools/submit_api.js check [--state 文件] [--user-id ID] [--problem-id ID]\n' +
      '       node tools/submit_api.js preview --file 源码.asc\n' +
      '       node tools/submit_api.js submit --file 源码.asc [--state 文件] [--user-id ID] [--problem-id ID]\n' +
      '路径可为绝对路径，或相对仓库根目录；提交时必须指定 --file。\n');
    return;
  }
  const source = ['submit', 'preview'].includes(options.action) ?
    selectedSource(options.file) : null;
  if (options.action === 'preview') {
    process.stdout.write(`${JSON.stringify({ sourcePath: source.path,
      kernelBytes: source.bytes, kernelSha256: source.sha256 }, null, 2)}\n`);
    return;
  }
  cookie = cookieFromState(options.state);
  userId = options.userId === 'auto' ? userIdFromState(options.state) : options.userId;
  const before = await getLatest();
  if (options.action === 'check') {
    process.stdout.write(`${JSON.stringify(before ? compact(before) :
      { status: '暂无提交', userId, problemId: options.problemId }, null, 2)}\n`);
    return;
  }

  const beforeStatus = String(before?.detail.status || before?.summary.status || '').toLowerCase();
  if (beforeStatus === 'running' || beforeStatus === 'pending') {
    throw new Error(`Latest submission ${before.summary.ID} is still ${beforeStatus}`);
  }

  const localKernel = source.content;

  // The submit API accepts only user-editable root .asc/.h files here. The
  // immutable problem template is injected by the judge and must not be sent.
  const editableFiles = [{ path: 'kernel.asc', content: localKernel }];

  const payload = {
    problemId: options.problemId,
    userId,
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
  if (!after || String(after.summary.ID) === String(before?.summary.ID)) {
    throw new Error(`Submit response did not create a new submission: ${JSON.stringify(response)}`);
  }
  const submitted = compact(after);
  if (submitted.kernelSha256 !== source.sha256 ||
      submitted.kernelBytes !== source.bytes) {
    throw new Error(`远端 kernel.asc 与所选源码不一致：本地 ${source.sha256}，远端 ${submitted.kernelSha256}`);
  }
  process.stdout.write(`${JSON.stringify({
    before: before?.summary.ID ?? null,
    after: after.summary.ID,
    id: after.summary._id,
    status: after.detail.status || after.summary.status,
    sourcePath: source.path,
    kernelBytes: source.bytes,
    kernelSha256: submitted.kernelSha256,
  }, null, 2)}\n`);
}

main().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
