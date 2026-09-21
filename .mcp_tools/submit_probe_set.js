async (page) => {
  const EDITOR = '.monaco-editor textarea.inputarea';
  const SUBMIT_URL = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';
  const modes = [3, 4, 5, 6, 7];
  const results = [];

  for (const mode of modes) {
    const file = `D:/cann_competition/.tmp_probes/kernel.p${mode}.asc`;
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.goto(SUBMIT_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector(EDITOR, { timeout: 40000 });
    await page.waitForTimeout(2000);

    await page.unroute('**/__local_kernel.asc');
    await page.route('**/__local_kernel.asc', route => route.fulfill({
      path: file, contentType: 'text/plain; charset=utf-8'
    }));
    const code = await page.evaluate(() =>
      fetch('/__local_kernel.asc', { cache: 'no-store' }).then(r => r.text()));
    const norm = code.replace(/\r\n/g, '\n');

    await page.evaluate(async (c) => { await navigator.clipboard.writeText(c); }, code);
    await page.evaluate((sel) => document.querySelector(sel).focus(), EDITOR);
    await page.keyboard.press('Control+a');
    await page.keyboard.press('Delete');
    await page.waitForTimeout(400);
    await page.keyboard.press('Control+v');
    await page.waitForTimeout(2500);

    let roundTrip = '';
    for (let attempt = 0; attempt < 6 && roundTrip.replace(/\r\n/g, '\n') !== norm; attempt++) {
      await page.evaluate(() => document.querySelector('button[title="复制当前文件"]').click());
      await page.waitForTimeout(800);
      roundTrip = await page.evaluate(() => navigator.clipboard.readText());
    }
    if (roundTrip.replace(/\r\n/g, '\n') !== norm) {
      results.push({ mode, ok: false, gotLen: roundTrip.length });
      continue;
    }

    await page.getByRole('button', { name: '提交代码' }).click();
    await page.waitForTimeout(4000);
    const dialog = await page.evaluate(() => {
      const d = document.querySelector('[role="dialog"]');
      return d ? d.innerText.replace(/\s+/g, ' ').trim().slice(0, 120) : null;
    });
    results.push({ mode, ok: true, dialog });
    await page.waitForTimeout(20000);  // submission rate limit
  }
  return results;
}
