const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..', '..');
const defaults = Object.freeze({
  state: '.tmp_cannjudge_state.json',
  userId: 'auto',
  problemId: '6a9aa054bf41025d6014f3ef',
});

function resolveFromRoot(value) {
  return path.resolve(root, value);
}

function parseArgs(argv, defaultAction, actions) {
  const options = { action: defaultAction, ...defaults };
  let sawAction = false;
  for (let index = 0; index < argv.length; ++index) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      options.help = true;
      continue;
    }
    if (!arg.startsWith('-') && !sawAction && actions.includes(arg)) {
      options.action = arg;
      sawAction = true;
      continue;
    }
    const match = /^(--file|--state|--user-id|--problem-id)(?:=(.*))?$/.exec(arg);
    if (!match) throw new Error(`未知参数：${arg}`);
    const value = match[2] === undefined ? argv[++index] : match[2];
    if (!value || value.startsWith('--')) throw new Error(`${match[1]} 缺少参数值`);
    const field = {
      '--file': 'file', '--state': 'state',
      '--user-id': 'userId', '--problem-id': 'problemId',
    }[match[1]];
    if (Object.hasOwn(options, `${field}Provided`)) {
      throw new Error(`重复参数：${match[1]}`);
    }
    options[field] = value;
    options[`${field}Provided`] = true;
  }
  if (!options.help && ['submit', 'preview'].includes(options.action) && !options.file) {
    throw new Error(`${options.action} 必须指定 --file <路径>；不会默认读取根目录 kernel.asc`);
  }
  if (!options.help && options.file && !['submit', 'preview'].includes(options.action)) {
    throw new Error('--file 只能与 preview 或 submit 一起使用');
  }
  return options;
}

function selectedSource(file) {
  const sourcePath = resolveFromRoot(file);
  if (path.extname(sourcePath).toLowerCase() !== '.asc') {
    throw new Error('提交源码必须是 .asc 文件');
  }
  if (!fs.statSync(sourcePath).isFile()) {
    throw new Error(`不是普通文件：${sourcePath}`);
  }
  const content = fs.readFileSync(sourcePath, 'utf8').replace(/\r\n/g, '\n');
  if (!content.trim()) throw new Error(`提交文件为空：${sourcePath}`);
  return {
    path: sourcePath,
    content,
    bytes: Buffer.byteLength(content),
    sha256: crypto.createHash('sha256').update(content).digest('hex'),
  };
}

function statePath(value) {
  const target = resolveFromRoot(value);
  if (!fs.statSync(target).isFile()) throw new Error(`不是登录状态文件：${target}`);
  return target;
}

function readState(value) {
  try {
    return JSON.parse(fs.readFileSync(statePath(value), 'utf8'));
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error('登录状态文件不是有效的 JSON');
    }
    throw error;
  }
}

function userIdFromState(value) {
  const state = readState(value);
  const origins = Array.isArray(state.origins) ? state.origins : [];
  for (const origin of origins) {
    if (!origin || !Array.isArray(origin.localStorage)) continue;
    const item = origin.localStorage.find(entry => entry.name === 'cannjudge_user');
    if (!item) continue;
    try {
      const user = JSON.parse(item.value);
      const objectId = user?._id ?? user?.id;
      if (typeof objectId === 'string' && objectId.trim()) return objectId.trim();
    } catch {
      throw new Error('登录状态中的 cannjudge_user 不是有效的 JSON');
    }
  }
  throw new Error('登录状态中没有当前账号 ID，请重新保存登录状态');
}

function cookieFromState(value) {
  const state = readState(value);
  if (!Array.isArray(state.cookies)) throw new Error('登录状态文件缺少 cookies 数组');
  const cookies = state.cookies.filter(item =>
    item.domain === 'cannjudge.cn' || item.domain.endsWith('.cannjudge.cn'));
  if (!cookies.length) throw new Error('登录状态文件没有 CANNJudge Cookie');
  return cookies.map(item => `${item.name}=${item.value}`).join('; ');
}

module.exports = {
  root, parseArgs, selectedSource, statePath, readState, userIdFromState, cookieFromState,
};
