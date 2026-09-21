async (page) => {
  const EDITOR = '.monaco-editor textarea.inputarea';
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit',
    { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'kernel.asc', exact: true }).first().click();
  await page.waitForSelector(EDITOR, { timeout: 40000 });
  await page.waitForTimeout(2500);

  // The page cannot reach 127.0.0.1 over its https origin, so the file is
  // injected through a route handler fulfilled by the Playwright process.
  await page.route('**/__local_kernel.asc', route => route.fulfill({
    path: require('path').resolve(process.cwd(), 'kernel.asc'),
    contentType: 'text/plain; charset=utf-8'
  }));
  const code = await page.evaluate(() =>
    fetch('/__local_kernel.asc', { cache: 'no-store' }).then(r => r.text()));
  const norm = code.replace(/\r\n/g, '\n');

  await page.evaluate((sel) => document.querySelector(sel).focus(), EDITOR);
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Delete');
  await page.waitForTimeout(500);
  await page.evaluate(({ selector, text }) => {
    const transfer = new DataTransfer();
    transfer.setData('text/plain', text);
    document.querySelector(selector).dispatchEvent(new ClipboardEvent('paste', {
      clipboardData: transfer,
      bubbles: true,
      cancelable: true,
    }));
  }, { selector: EDITOR, text: code });
  await page.waitForTimeout(3000);

  // Read the editor back through the app's own "copy current file" action.
  const roundTrip = await page.evaluate(async () => {
    window.__capturedKernelCopy = null;
    Object.defineProperty(navigator.clipboard, 'writeText', {
      configurable: true,
      value: async text => { window.__capturedKernelCopy = text; },
    });
    document.querySelector('button[title="复制当前文件"]').click();
    await new Promise(resolve => setTimeout(resolve, 700));
    return window.__capturedKernelCopy || '';
  });
  if (roundTrip.replace(/\r\n/g, '\n') !== norm) {
    const shown = await page.evaluate(() => {
      const ls = [...document.querySelectorAll('.view-lines .view-line')].slice(0, 3);
      return ls.map(e => e.textContent.slice(0, 60));
    });
    return { ok: false, reason: 'editor content mismatch',
             srcLen: norm.length, gotLen: roundTrip.length, shown };
  }

  await page.getByRole('button', { name: '提交代码' }).click();
  await page.waitForTimeout(6000);
  const dialog = await page.evaluate(() => {
    const d = document.querySelector('[role="dialog"]');
    return d ? d.innerText.replace(/\s+/g, ' ').trim().slice(0, 200) : null;
  });
  const m = page.url().match(/submission\/([0-9a-f]+)/);
  return { ok: true, contentLen: norm.length, submissionId: m ? m[1] : null, dialog };
}
