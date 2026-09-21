async (page) => {
  await page.route('**/__current_kernel.asc', route => route.fulfill({
    path: require('path').resolve(process.cwd(), 'kernel.asc'),
    contentType: 'text/plain; charset=utf-8',
  }));
  const current = await page.evaluate(() =>
    fetch('/__current_kernel.asc', { cache: 'no-store' }).then(r => r.text()));
  const passed = await page.evaluate(async () => {
    const value = await (await fetch(
      '/api/submissions/6aaf71e3b0477ec41e4f0fcc',
      { credentials: 'include' })).json();
    return value.files.find(file => file.path === 'kernel.asc').content;
  });
  const a = passed.replace(/\r\n/g, '\n').split('\n');
  const b = current.replace(/\r\n/g, '\n').split('\n');
  const diffs = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; ++i) {
    if (a[i] !== b[i]) diffs.push({ line: i + 1, passed: a[i], current: b[i] });
  }
  return { passedLines: a.length, currentLines: b.length, diffs };
}
