const fs = require('fs');
const path = require('path');
const readline = require('readline/promises');
const { stdin, stdout } = require('process');
const { parseArgs, root } = require('./lib/cli');

const options = parseArgs(process.argv.slice(2), 'save', ['save']);
const url = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';

async function main() {
  if (options.help) {
    process.stdout.write('用法：node tools/save_login_state.js [--state 文件]\n' +
      '脚本会打开可见浏览器。请自行登录，然后在终端按 Enter。\n' +
      '默认输出：仓库根目录的 .tmp_cannjudge_state.json。\n');
    return;
  }
  const output = path.resolve(root, options.state);
  const { chromium } = require('playwright');
  const browser = await chromium.launch({
    headless: false,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
      : {}),
  });
  const prompt = readline.createInterface({ input: stdin, output: stdout });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await prompt.question('请在浏览器中登录 CANNJudge，完成后在此按 Enter 保存登录状态。');
    const cookies = await context.cookies();
    const judgeCookies = cookies.filter(item =>
      item.domain === 'cannjudge.cn' || item.domain.endsWith('.cannjudge.cn'));
    if (!judgeCookies.length) {
      const domains = [...new Set(cookies.map(item => item.domain))].join(', ') || '无';
      throw new Error(`未发现 CANNJudge Cookie，检测到的域名：${domains}。请确认是在本脚本打开的浏览器中完成登录。`);
    }
    if (fs.existsSync(output)) {
      const answer = await prompt.question(`覆盖已有登录状态文件 ${output}？请输入 yes：`);
      if (answer.trim() !== 'yes') throw new Error('已保留原登录状态文件');
    }
    fs.mkdirSync(path.dirname(output), { recursive: true });
    await context.storageState({ path: output });
    if (process.platform !== 'win32') fs.chmodSync(output, 0o600);
    process.stdout.write(`登录状态已保存到 ${output}。请妥善保管此文件。\n`);
  } finally {
    prompt.close();
    await browser.close();
  }
}

main().catch(error => {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
});
