const crypto = require('crypto');
const fs = require('fs');

const USER_ID = '6aabe4f6b0477ec41ec44882';
const PROBLEM_ID = '6a9aa054bf41025d6014f3ef';
const state = JSON.parse(fs.readFileSync('.tmp_cannjudge_state.json', 'utf8'));
const cookie = state.cookies
  .filter(item => item.domain === 'cannjudge.cn' || item.domain === '.cannjudge.cn')
  .map(item => `${item.name}=${item.value}`)
  .join('; ');

async function get(path) {
  const response = await fetch(`https://cannjudge.cn${path}`, {
    headers: { cookie, accept: 'application/json' },
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
  return response.json();
}

function normalize(source) {
  return String(source)
    .replace(/\r\n/g, '\n')
    .replace(/#define\s+BMMS_PROBE_MODE\s+\d+/, '#define BMMS_PROBE_MODE 0');
}

(async () => {
  const list = await get(`/api/submissions/user/${USER_ID}/problem/${PROBLEM_ID}`);
  const rows = [];
  for (const item of list) {
    const detail = await get(`/api/submissions/${item._id}`);
    const files = typeof detail.files === 'string' ? JSON.parse(detail.files) : detail.files;
    const source = files?.find(file => file.path === 'kernel.asc')?.content || '';
    if (!source.includes('BMMS_PROBE_MODE')) continue;
    const mode = Number(source.match(/#define\s+BMMS_PROBE_MODE\s+(\d+)/)?.[1] || 0);
    const hash = crypto.createHash('sha256').update(normalize(source)).digest('hex').slice(0, 16);
    rows.push({
      ID: item.ID,
      objectId: item._id,
      hash,
      mode,
      status: detail.status,
      times: (detail.result || []).map(result => result.time),
      precision: (detail.result || []).map(result => result.precision_ratio),
    });
  }
  const probeHashes = new Set(rows.filter(row => row.mode !== 0).map(row => row.hash));
  const relevant = rows.filter(row => probeHashes.has(row.hash));
  const groups = Object.groupBy
    ? Object.groupBy(relevant, row => row.hash)
    : relevant.reduce((out, row) => {
        (out[row.hash] ||= []).push(row);
        return out;
      }, {});
  process.stdout.write(`${JSON.stringify(groups, null, 2)}\n`);
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
