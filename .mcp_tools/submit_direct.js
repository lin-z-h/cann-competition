async (page) => {
  const EDITOR = '.monaco-editor textarea.inputarea';
  const URL = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';
  const uid = '6aabe4f6b0477ec41ec44882';
  const pid = '6a9aa054bf41025d6014f3ef';

  const digest = async (text) => {
    const bytes = new TextEncoder().encode(text);
    const hash = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join('');
  };
  const latest = () => page.evaluate(async ({ uid, pid }) => {
    const list = await (await fetch(`/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    return { id: list[0]._id, ID: list[0].ID };
  }, { uid, pid });

  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector(EDITOR, { timeout: 40000 });
  await page.waitForTimeout(2000);
  await page.route('**/__local_kernel.asc', route => route.fulfill({
    path: 'D:/cann_competition/kernel.asc',
    contentType: 'text/plain; charset=utf-8',
  }));
  const code = await page.evaluate(() =>
    fetch('/__local_kernel.asc', { cache: 'no-store' }).then(r => r.text()));
  const normalized = code.replace(/\r\n/g, '\n');
  const expectedHash = await page.evaluate(digest, normalized);
  const before = await latest();

  await page.locator(EDITOR).focus();
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Delete');
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

  const captured = await page.evaluate(async () => {
    const clipboard = navigator.clipboard;
    window.__capturedCopy = null;
    const original = clipboard.writeText.bind(clipboard);
    Object.defineProperty(clipboard, 'writeText', {
      configurable: true,
      value: async (text) => {
        window.__capturedCopy = text;
        return original(text);
      },
    });
    document.querySelector('button[title="复制当前文件"]').click();
    await new Promise(resolve => setTimeout(resolve, 700));
    return window.__capturedCopy;
  });
  const actual = (captured ?? '').replace(/\r\n/g, '\n');
  const actualHash = await page.evaluate(digest, actual);
  if (actual !== normalized) {
    return { ok: false, reason: 'editor content mismatch', expectedLen: normalized.length,
      actualLen: actual.length, expectedHash, actualHash };
  }

  await page.getByRole('button', { name: '提交代码' }).click();
  await page.waitForTimeout(6000);
  const dialog = await page.evaluate(() => {
    const node = document.querySelector('[role="dialog"]');
    return node ? node.innerText.replace(/\s+/g, ' ').trim() : null;
  });
  const after = await latest();
  return { ok: after.ID !== before.ID, before: before.ID, after: after.ID,
    submissionId: after.id, expectedLen: normalized.length, expectedHash, actualHash, dialog };
}
