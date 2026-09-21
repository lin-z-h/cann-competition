const fs = require('fs');

const submissionId = process.argv[2];
if (!submissionId) throw new Error('Usage: node inspect_detail_api.js <submission-object-id>');

const state = JSON.parse(fs.readFileSync('.tmp_cannjudge_state.json', 'utf8'));
const cookie = state.cookies
  .filter(item => item.domain === 'cannjudge.cn' || item.domain === '.cannjudge.cn')
  .map(item => `${item.name}=${item.value}`)
  .join('; ');

(async () => {
  const response = await fetch(`https://cannjudge.cn/api/submissions/${submissionId}`, {
    headers: { cookie, accept: 'application/json' },
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const detail = await response.json();
  const out = { ...detail };
  if (typeof out.files === 'string') out.files = JSON.parse(out.files);
  if (Array.isArray(out.files)) {
    out.files = out.files.map(file => ({
      path: file.path,
      bytes: typeof file.content === 'string' ? Buffer.byteLength(file.content) : 0,
      message: file.message,
    }));
  }
  delete out.code;
  process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
