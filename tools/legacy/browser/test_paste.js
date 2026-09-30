async (page) => {
  const url = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';
  const editor = '.monaco-editor textarea.inputarea';
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector(editor, { timeout: 40000 });
  await page.waitForTimeout(1500);
  await page.locator(editor).focus();
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Delete');
  const pasted = await page.evaluate((selector) => {
    const transfer = new DataTransfer();
    transfer.setData('text/plain', 'abc\nxyz');
    const event = new ClipboardEvent('paste', {
      clipboardData: transfer,
      bubbles: true,
      cancelable: true,
    });
    return document.querySelector(selector).dispatchEvent(event);
  }, editor);
  await page.waitForTimeout(500);
  const captured = await page.evaluate(async () => {
    const clipboard = navigator.clipboard;
    window.__capturedPasteTest = null;
    const original = clipboard.writeText.bind(clipboard);
    Object.defineProperty(clipboard, 'writeText', {
      configurable: true,
      value: async text => {
        window.__capturedPasteTest = text;
        return original(text);
      },
    });
    document.querySelector('button[title="复制当前文件"]').click();
    await new Promise(resolve => setTimeout(resolve, 500));
    return window.__capturedPasteTest;
  });
  return { pasted, captured };
}
