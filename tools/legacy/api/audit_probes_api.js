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

(async () => {
  const list = await get(`/api/submissions/user/${USER_ID}/problem/${PROBLEM_ID}`);
  const found = [];
  for (const item of list) {
    const detail = await get(`/api/submissions/${item._id}`);
    const files = typeof detail.files === 'string' ? JSON.parse(detail.files) : detail.files;
    const source = files?.find(file => file.path === 'kernel.asc')?.content || '';
    const mode = Number(source.match(/#define\s+BMMS_PROBE_MODE\s+(\d+)/)?.[1] || 0);
    if (!mode) continue;
    found.push({
      ID: item.ID,
      objectId: item._id,
      mode,
      status: detail.status,
      probeClause: source.match(new RegExp(
        `#(?:if|elif) BMMS_PROBE_MODE == ${mode}[^#]+`))?.[0]
        ?.replace(/\s+/g, ' ').trim(),
      times: (detail.result || []).map(result => result.time),
      precision: (detail.result || []).map(result => result.precision_ratio),
    });
  }
  process.stdout.write(`${JSON.stringify({ submissions: list.length, probes: found }, null, 2)}\n`);
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
