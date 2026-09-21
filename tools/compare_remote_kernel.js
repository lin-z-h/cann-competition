const fs = require('fs');

const submissionId = process.argv[2];
if (!submissionId) throw new Error('Usage: node compare_remote_kernel.js <submission-object-id>');
const secondSubmissionId = process.argv[3];

const state = JSON.parse(fs.readFileSync('.tmp_cannjudge_state.json', 'utf8'));
const cookie = state.cookies
  .filter(item => item.domain === 'cannjudge.cn' || item.domain === '.cannjudge.cn')
  .map(item => `${item.name}=${item.value}`)
  .join('; ');

function lines(text) {
  return String(text).replace(/\r\n/g, '\n').split('\n');
}

function unifiedDiff(oldText, newText) {
  const a = lines(oldText);
  const b = lines(newText);
  const width = b.length + 1;
  const table = new Uint16Array((a.length + 1) * width);
  for (let i = a.length - 1; i >= 0; --i) {
    for (let j = b.length - 1; j >= 0; --j) {
      table[i * width + j] = a[i] === b[j]
        ? table[(i + 1) * width + j + 1] + 1
        : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      ops.push({ type: ' ', text: a[i], oldLine: i + 1, newLine: j + 1 });
      ++i;
      ++j;
    } else if (j < b.length && (i === a.length ||
        table[i * width + j + 1] >= table[(i + 1) * width + j])) {
      ops.push({ type: '+', text: b[j], oldLine: i + 1, newLine: j + 1 });
      ++j;
    } else {
      ops.push({ type: '-', text: a[i], oldLine: i + 1, newLine: j + 1 });
      ++i;
    }
  }
  const changed = ops.map((op, index) => op.type === ' ' ? -1 : index).filter(index => index >= 0);
  const selected = new Set();
  for (const index of changed) {
    for (let at = Math.max(0, index - 3); at <= Math.min(ops.length - 1, index + 3); ++at) {
      selected.add(at);
    }
  }
  const output = ['--- remote-pass', '+++ local-current'];
  let previous = -2;
  for (const index of [...selected].sort((x, y) => x - y)) {
    if (index > previous + 1) output.push('@@');
    const op = ops[index];
    output.push(`${op.type}${String(op.oldLine).padStart(4)}:${String(op.newLine).padStart(4)} ${op.text}`);
    previous = index;
  }
  return { output: output.join('\n'), changedLines: changed.length };
}

(async () => {
  async function loadRemote(id) {
    const response = await fetch(`https://cannjudge.cn/api/submissions/${id}`, {
      headers: { cookie, accept: 'application/json' },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const detail = await response.json();
    const files = typeof detail.files === 'string' ? JSON.parse(detail.files) : detail.files;
    const remote = files.find(file => file.path === 'kernel.asc');
    if (!remote) throw new Error('Remote kernel.asc missing');
    return remote.content;
  }
  const first = await loadRemote(submissionId);
  const second = secondSubmissionId
    ? await loadRemote(secondSubmissionId)
    : fs.readFileSync('kernel.asc', 'utf8');
  const diff = unifiedDiff(first, second);
  process.stdout.write(`changedLines=${diff.changedLines}\n${diff.output}\n`);
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
