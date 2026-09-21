async (page) => {
  const selector = '.monaco-editor textarea.inputarea';
  await page.goto(
    'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit',
    { waitUntil: 'domcontentloaded' });
  await page.waitForSelector(selector, { timeout: 40000 });
  await page.waitForTimeout(2000);
  await page.route('**/__local_kernel_reset.asc', route => route.fulfill({
    path: 'D:/cann_competition/kernel.asc',
    contentType: 'text/plain; charset=utf-8',
  }));
  const code = await page.evaluate(() =>
    fetch('/__local_kernel_reset.asc', { cache: 'no-store' }).then(r => r.text()));
  const area = page.locator(selector);
  await area.focus();
  await page.keyboard.press('Control+a');
  await page.waitForTimeout(300);
  await page.keyboard.press('Backspace');
  await page.waitForTimeout(500);
  await page.evaluate(({ selector, code }) => {
    const transfer = new DataTransfer();
    transfer.setData('text/plain', code);
    document.querySelector(selector).dispatchEvent(new ClipboardEvent('paste', {
      clipboardData: transfer, bubbles: true, cancelable: true,
    }));
  }, { selector, code });
  await page.waitForTimeout(2500);
  const captured = await page.evaluate(async () => {
    window.__resetCapture = null;
    const clipboard = navigator.clipboard;
    Object.defineProperty(clipboard, 'writeText', {
      configurable: true,
      value: async text => { window.__resetCapture = text; },
    });
    document.querySelector('button[title="复制当前文件"]').click();
    await new Promise(resolve => setTimeout(resolve, 700));
    return window.__resetCapture;
  });
  return {
    ok: captured?.replace(/\r\n/g, '\n') === code.replace(/\r\n/g, '\n'),
    expectedLen: code.replace(/\r\n/g, '\n').length,
    actualLen: captured?.replace(/\r\n/g, '\n').length,
  };
}
