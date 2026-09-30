const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { root, parseArgs, selectedSource, statePath } = require('./lib/cli');

const browserTemp = path.join(root, '.tmp_playwright_runtime');
const options = parseArgs(process.argv.slice(2), 'poll', ['poll', 'preview', 'submit']);
const submitUrl =
  'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';

function chromiumExecutable() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  const cacheRoot = process.env.PLAYWRIGHT_BROWSERS_PATH &&
    process.env.PLAYWRIGHT_BROWSERS_PATH !== '0'
    ? process.env.PLAYWRIGHT_BROWSERS_PATH
    : path.join(process.env.LOCALAPPDATA || '', 'ms-playwright');
  if (!cacheRoot || !fs.existsSync(cacheRoot)) return undefined;
  const candidates = fs.readdirSync(cacheRoot)
    .filter(name => /^chromium-\d+$/.test(name))
    .sort()
    .reverse()
    .map(name => path.join(cacheRoot, name, 'chrome-win64', 'chrome.exe'))
    .filter(candidate => fs.existsSync(candidate));
  return candidates[0];
}

async function resolveUserId(page) {
  if (options.userId !== 'auto') return options.userId;
  return page.evaluate(() => {
    const raw = localStorage.getItem('cannjudge_user');
    if (!raw) return '';
    try {
      const user = JSON.parse(raw);
      return String(user?._id ?? user?.id ?? '');
    } catch {
      return '';
    }
  });
}

async function latest(page) {
  const uid = await resolveUserId(page);
  if (!uid) throw new Error('登录状态中没有当前账号 ID，请重新保存登录状态');
  return page.evaluate(async ({ uid, pid }) => {
    const list = await (await fetch(
      `/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    return list.length ? { id: list[0]._id, ID: list[0].ID } :
      { id: null, ID: null };
  }, { uid, pid: options.problemId });
}

async function poll(page) {
  const current = await latest(page);
  if (!current.id) return { status: '暂无提交' };
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

async function submit(page, source) {
  const editor = '.monaco-editor textarea.inputarea';
  const code = source.content;
  const normalized = code;
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
  if (!after.id || after.ID === before.ID) {
    throw new Error('提交后未找到新的 submission');
  }
  const remoteKernel = await page.evaluate(async id => {
    const response = await fetch(`/api/submissions/${id}`, {
      credentials: 'include',
    });
    if (!response.ok) throw new Error(`读取提交详情失败：HTTP ${response.status}`);
    const detail = await response.json();
    const files = typeof detail.files === 'string' ?
      JSON.parse(detail.files) : detail.files;
    return Array.isArray(files) ?
      files.find(file => file.path === 'kernel.asc')?.content ?? null : null;
  }, after.id);
  const remoteContent = typeof remoteKernel === 'string' ?
    remoteKernel.replace(/\r\n/g, '\n') : '';
  const remoteHash = remoteContent ?
    crypto.createHash('sha256').update(remoteContent).digest('hex') : null;
  if (remoteHash !== source.sha256 ||
      Buffer.byteLength(remoteContent) !== source.bytes) {
    throw new Error(`远端 kernel.asc 与所选源码不一致：本地 ${source.sha256}，远端 ${remoteHash}`);
  }
  const dialog = await page.locator('[role="dialog"]').count()
    ? await page.locator('[role="dialog"]').innerText() : '';
  return { ok: true, before: before.ID, after,
    sourcePath: source.path, kernelSha256: remoteHash, dialog };
}

(async () => {
  if (options.help) {
    process.stdout.write('用法：node tools/standalone_browser.js poll [--state 文件] [--user-id ID] [--problem-id ID]\n' +
      '       node tools/standalone_browser.js preview --file 源码.asc\n' +
      '       node tools/standalone_browser.js submit --file 源码.asc [--state 文件] [--user-id ID] [--problem-id ID]\n' +
      '路径可为绝对路径，或相对仓库根目录；提交时必须指定 --file。\n');
    return;
  }
  const source = ['submit', 'preview'].includes(options.action) ?
    selectedSource(options.file) : null;
  if (options.action === 'preview') {
    process.stdout.write(`${JSON.stringify({ sourcePath: source.path,
      kernelBytes: source.bytes, kernelSha256: source.sha256 }, null, 2)}\n`);
    return;
  }
  const loginState = statePath(options.state);
  fs.mkdirSync(browserTemp, { recursive: true });
  process.env.TEMP = browserTemp;
  process.env.TMP = browserTemp;
  const { chromium } = require('playwright');
  const browser = await chromium.launch({
    headless: true,
    ...(chromiumExecutable() ? { executablePath: chromiumExecutable() } : {}),
  });
  try {
    const context = await browser.newContext({ storageState: loginState });
    const page = await context.newPage();
    await page.goto(submitUrl, { waitUntil: 'domcontentloaded' });
    const loggedIn = await page.evaluate(() => Boolean(
      localStorage.getItem('cannjudge_user')));
    if (!loggedIn) throw new Error('saved login state expired');
    const result = options.action === 'submit' ? await submit(page, source) : await poll(page);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
