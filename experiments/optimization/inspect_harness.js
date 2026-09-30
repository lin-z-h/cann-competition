// Read-only inspection of the judge template shipped with a submission.
const { cookieFromState } = require('../../tools/lib/cli');

(async () => {
  const response = await fetch(
    'https://cannjudge.cn/api/submissions/6ab229be0304f72a5647e5ae',
    { headers: { accept: 'application/json', cookie: cookieFromState('.tmp_cannjudge_state.json') } });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const detail = await response.json();
  const files = typeof detail.files === 'string' ? JSON.parse(detail.files) : detail.files;
  for (const file of files || []) {
    if (!/^(run\.sh|scripts\/|main\.asc$)/.test(file.path)) continue;
    const lines = String(file.content || '').split(/\r?\n/);
    for (let index = 0; index < lines.length; index++) {
      if (/profil|tim(e|ing)|benchmark|latency|aclrtSynchronize|run_kernel|warmup/i.test(lines[index])) {
        process.stdout.write(`${file.path}:${index + 1}: ${lines[index].trim()}\n`);
      }
    }
  }
})().catch(error => { process.stderr.write(`${error.message}\n`); process.exit(1); });
