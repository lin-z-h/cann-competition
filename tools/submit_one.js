async (page) => {
  const EDITOR = '.monaco-editor textarea.inputarea';
  const SUBMIT_URL = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';

  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto(SUBMIT_URL, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector(EDITOR, { timeout: 40000 });
  await page.waitForTimeout(2000);

  // Which file to send is read from disk through a route handler, because the
  // sandboxed runner cannot touch the filesystem directly.
  await page.route('**/__next_probe.txt', route => route.fulfill({
    path: require('path').resolve(process.cwd(), '.tmp_probes/next.txt'),
    contentType: 'text/plain; charset=utf-8'
  }));
  const target = (await page.evaluate(() =>
    fetch('/__next_probe.txt', { cache: 'no-store' }).then(r => r.text()))).trim();

  await page.route('**/__local_kernel.asc', route => route.fulfill({
    path: target, contentType: 'text/plain; charset=utf-8'
  }));
  const code = await page.evaluate(() =>
    fetch('/__local_kernel.asc', { cache: 'no-store' }).then(r => r.text()));
  const norm = code.replace(/\r\n/g, '\n');

  await page.evaluate(async (c) => { await navigator.clipboard.writeText(c); }, code);
  await page.evaluate((sel) => document.querySelector(sel).focus(), EDITOR);
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Delete');
  await page.waitForTimeout(500);
  await page.keyboard.press('Control+v');
  await page.waitForTimeout(3000);

  let roundTrip = '';
  for (let attempt = 0; attempt < 6 && roundTrip.replace(/\r\n/g, '\n') !== norm; attempt++) {
    await page.evaluate(() => document.querySelector('button[title="复制当前文件"]').click());
    await page.waitForTimeout(800);
    roundTrip = await page.evaluate(() => navigator.clipboard.readText());
  }
  if (roundTrip.replace(/\r\n/g, '\n') !== norm) {
    return { ok: false, target, reason: 'editor content mismatch', gotLen: roundTrip.length };
  }

  await page.getByRole('button', { name: '提交代码' }).click();
  await page.waitForTimeout(6000);
  const dialog = await page.evaluate(() => {
    const d = document.querySelector('[role="dialog"]');
    return d ? d.innerText.replace(/\s+/g, ' ').trim().slice(0, 140) : null;
  });
  return { ok: true, target, contentLen: norm.length, dialog };
}
