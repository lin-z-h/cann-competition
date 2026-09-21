const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const browserTemp = path.join(root, '.tmp_playwright_runtime');
fs.mkdirSync(browserTemp, { recursive: true });
process.env.TEMP = browserTemp;
process.env.TMP = browserTemp;
const statePath = path.join(root, '.tmp_cannjudge_state.json');
const kernelPath = path.join(root, 'kernel.asc');
const uid = '6aabe4f6b0477ec41ec44882';
const pid = '6a9aa054bf41025d6014f3ef';
const submitUrl =
  'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';

async function latest(page) {
  return page.evaluate(async ({ uid, pid }) => {
    const list = await (await fetch(
      `/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    return { id: list[0]._id, ID: list[0].ID };
  }, { uid, pid });
}

async function poll(page) {
  const current = await latest(page);
  return page.evaluate(async id => {
    const data = await (await fetch(`/api/submissions/${id}`, {
      credentials: 'include',
    })).json();
    return {
      ID: data.ID,
      status: data.status,
      precision: (data.result || []).map(x => x.precision_ratio),
      times: (data.result || []).map(x => x.time),
      best: (data.result || []).map(x => x.best_time),
    };
  }, current.id);
}

async function submit(page) {
  const editor = '.monaco-editor textarea.inputarea';
  const code = fs.readFileSync(kernelPath, 'utf8');
  const normalized = code.replace(/\r\n/g, '\n');
  await page.goto(submitUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector(editor, { timeout: 40000 });
  const before = await latest(page);
  await page.locator(editor).focus();
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
  }, { selector: editor, text: code });
  await page.waitForTimeout(2500);
  const actual = await page.evaluate(() => {
    const model = window.monaco?.editor?.getModels?.()[0];
    return model ? model.getValue() : null;
  });
  if ((actual || '').replace(/\r\n/g, '\n') !== normalized) {
    throw new Error('editor content mismatch');
  }
  await page.getByRole('button', { name: '提交代码' }).click();
  await page.waitForTimeout(6000);
  const after = await latest(page);
  const dialog = await page.locator('[role="dialog"]').count()
    ? await page.locator('[role="dialog"]').innerText() : '';
  return { ok: after.ID !== before.ID, before: before.ID, after, dialog };
}

(async () => {
  const action = process.argv[2] || 'poll';
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
      : {}),
  });
  try {
    const context = await browser.newContext({ storageState: statePath });
    const page = await context.newPage();
    await page.goto(submitUrl, { waitUntil: 'domcontentloaded' });
    const loggedIn = await page.getByText('退出', { exact: true }).count();
    if (!loggedIn) throw new Error('saved login state expired');
    const result = action === 'submit' ? await submit(page) : await poll(page);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
