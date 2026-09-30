const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const {
  root, parseArgs, selectedSource, cookieFromState, userIdFromState,
} = require('../lib/cli');

test('提交必须显式指定源码文件', () => {
  assert.throws(() => parseArgs(['submit'], 'check', ['check', 'submit']), /必须指定 --file/);
  assert.throws(() => parseArgs(['check', '--file', 'kernel.asc'], 'check', ['check', 'submit']),
    /只能与 preview 或 submit/);
  const options = parseArgs(
    ['submit', '--file', 'experiments/new_design/kernel.asc', '--state=custom.json'],
    'check', ['check', 'submit']);
  assert.equal(options.file, 'experiments/new_design/kernel.asc');
  assert.equal(options.state, 'custom.json');
  assert.equal(parseArgs(['preview', '--file', 'kernel.asc'], 'check',
    ['check', 'preview', 'submit']).action, 'preview');
  assert.throws(() => parseArgs(['preview'], 'check', ['check', 'preview', 'submit']),
    /必须指定 --file/);
});

test('按指定路径选择源码且不改动可靠版本', () => {
  const source = selectedSource('experiments/new_design/kernel.asc');
  assert.equal(source.path, path.join(root, 'experiments/new_design/kernel.asc'));
  assert.notEqual(source.content, fs.readFileSync(path.join(root, 'kernel.asc'), 'utf8'));
  assert.equal(source.sha256,
    crypto.createHash('sha256').update(source.content).digest('hex'));
  assert.equal(selectedSource(source.path).sha256, source.sha256);
  assert.throws(() => selectedSource('tools/README.md'), /\.asc 文件/);
});

test('只读取 CANNJudge 域名的 Cookie', () => {
  const fixture = path.join(root, `.tmp_cli_state_${process.pid}.json`);
  fs.writeFileSync(fixture, JSON.stringify({ cookies: [
    { domain: '.cannjudge.cn', name: 'session', value: 'dummy' },
    { domain: 'other.example', name: 'unrelated', value: 'skip' },
  ] }));
  try {
    assert.equal(cookieFromState(fixture), 'session=dummy');
  } finally {
    fs.unlinkSync(fixture);
  }
});

test('从登录状态自动读取当前账号对象 ID', () => {
  const fixture = path.join(root, `.tmp_cli_user_${process.pid}.json`);
  fs.writeFileSync(fixture, JSON.stringify({
    cookies: [{ domain: 'cannjudge.cn', name: 'session', value: 'dummy' }],
    origins: [{ origin: 'https://cannjudge.cn', localStorage: [{
      name: 'cannjudge_user',
      value: JSON.stringify({ _id: '6ab152d00304f72a56ecde87', ID: 7009 }),
    }] }],
  }));
  try {
    assert.equal(userIdFromState(fixture), '6ab152d00304f72a56ecde87');
  } finally {
    fs.unlinkSync(fixture);
  }
});
