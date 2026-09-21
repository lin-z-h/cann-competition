const crypto = require('crypto');
const fs = require('fs');

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

(async () => {
  const user = '6aabe4f6b0477ec41ec44882';
  const problem = '6a9aa054bf41025d6014f3ef';
  const list = await get(`/api/submissions/user/${user}/problem/${problem}`);
  const rows = [];
  for (const item of list) {
    const detail = await get(`/api/submissions/${item._id}`);
    if (detail.status !== 'Pass') continue;
    const files = typeof detail.files === 'string' ? JSON.parse(detail.files) : detail.files;
    const source = String(files?.find(file => file.path === 'kernel.asc')?.content || '')
      .replace(/\r\n/g, '\n');
    rows.push({
      ID: item.ID,
      objectId: item._id,
      hash: crypto.createHash('sha256').update(source).digest('hex').slice(0, 12),
      bytes: Buffer.byteLength(source),
      probe: Number(source.match(/#define\s+BMMS_PROBE_MODE\s+(\d+)/)?.[1] || 0),
      scalar: source.includes('ProcessScalarBatch'),
      async: source.includes('Iterate<false>'),
      nParallel: source.includes('ProcessNParallel'),
      times: (detail.result || []).map(result => result.time),
    });
  }
  process.stdout.write(`${JSON.stringify(rows, null, 2)}\n`);
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
