const fs = require('fs');

const state = JSON.parse(fs.readFileSync('.tmp_cannjudge_state.json', 'utf8'));
const cookie = state.cookies
  .filter(item => item.domain === 'cannjudge.cn' || item.domain === '.cannjudge.cn')
  .map(item => `${item.name}=${item.value}`)
  .join('; ');

(async () => {
  const url =
    'https://cannjudge.cn/public/op_challenge_shanghe_prelim/' +
    'batchmatmulmaxsum/submit';
  const response = await fetch(url, { headers: { cookie } });
  const html = await response.text();
  const scripts = [...html.matchAll(/<script[^>]+src="([^"]+)"/g)]
    .map(match => new URL(match[1], url).href);
  const matches = [];
  for (const script of scripts) {
    const text = await (await fetch(script, { headers: { cookie } })).text();
    const assetUrls = [...text.matchAll(/["'`]([^"'`]+\.js)["'`]/g)]
      .map(match => new URL(match[1], script).href);
    for (const assetUrl of assetUrls) {
      if (!scripts.includes(assetUrl)) scripts.push(assetUrl);
    }
    if (/api\/submissions|提交代码|submit/i.test(text)) {
      const snippets = [];
      for (const pattern of [
        /\/api\/submissions[^"'`\\ ]*/g,
        /fetch\([^)]{0,500}/g,
        /axios\.[a-z]+\([^)]{0,500}/g,
      ]) {
        const found = text.match(pattern) || [];
        snippets.push(...found.slice(0, 20));
      }
      matches.push({ script, length: text.length, snippets });
      const submitIndex = text.indexOf('/api/submissions/submit');
      if (submitIndex >= 0) {
        matches[matches.length - 1].submitContext = text.slice(
          Math.max(0, submitIndex - 2500), submitIndex + 2500);
      }
      const collectIndex = text.indexOf('collectPayload');
      if (collectIndex >= 0) {
        matches[matches.length - 1].collectContext = text.slice(
          Math.max(0, collectIndex - 2500), collectIndex + 3500);
      }
      const legacyIndex = text.indexOf('projectFilesToLegacyPayload');
      if (legacyIndex >= 0) {
        matches[matches.length - 1].legacyContext = text.slice(
          Math.max(0, legacyIndex - 1000), legacyIndex + 3000);
      }
    }
  }
  process.stdout.write(`${JSON.stringify({ status: response.status, scripts, matches })}\n`);
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
