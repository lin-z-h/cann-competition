async (page) => {
  const EDITOR = '.monaco-editor textarea.inputarea';
  const URL = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';
  const out = [];

  const latest = () => page.evaluate(async () => {
    const uid = '6aabe4f6b0477ec41ec44882';
    const pid = '6a9aa054bf41025d6014f3ef';
    const list = await (await fetch(`/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    const j = await (await fetch(`/api/submissions/${list[0]._id}`,
      { credentials: 'include' })).json();
    const res = j.result || [];
    return {
      ID: list[0].ID, status: j.status, cases: res.length,
      prec: res.map(x => Number(x.precision_ratio).toFixed(2)).join(','),
      sec: res.map(x => x.time).join(','),
    };
  });

  const loadInto = async (file) => {
    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector(EDITOR, { timeout: 40000 });
    await page.waitForTimeout(2500);
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.unroute('**/__local_kernel.asc');
    await page.route('**/__local_kernel.asc', route => route.fulfill({
      path: file, contentType: 'text/plain; charset=utf-8'
    }));
    const code = await page.evaluate(() =>
      fetch('/__local_kernel.asc', { cache: 'no-store' }).then(r => r.text()));
    const norm = code.replace(/\r\n/g, '\n');
    for (let attempt = 0; attempt < 4; attempt++) {
      await page.evaluate(async (c) => { await navigator.clipboard.writeText(c); }, code);
      await page.evaluate((sel) => document.querySelector(sel).focus(), EDITOR);
      await page.keyboard.press('Control+a');
      await page.keyboard.press('Delete');
      await page.waitForTimeout(600);
      await page.keyboard.press('Control+v');
      await page.waitForTimeout(2500 + attempt * 1500);
      let rt = '';
      for (let i = 0; i < 5 && rt.replace(/\r\n/g, '\n') !== norm; i++) {
        await page.evaluate(() => document.querySelector('button[title="复制当前文件"]').click());
        await page.waitForTimeout(900);
        rt = await page.evaluate(() => navigator.clipboard.readText());
      }
      if (rt.replace(/\r\n/g, '\n') === norm) return true;
    }
    return false;
  };

  const doSubmit = async () => {
    const before = (await latest()).ID;
    for (let t = 0; t < 12; t++) {
      try {
        await page.getByRole('button', { name: '提交代码' }).click({ timeout: 15000 });
      } catch (e) {
        await page.waitForTimeout(5000);
        continue;
      }
      await page.waitForTimeout(4000);
      const dlg = await page.evaluate(() => {
        const d = document.querySelector('[role="dialog"]');
        return d ? d.innerText.replace(/\s+/g, ' ').trim() : null;
      });
      if (dlg && dlg.includes('频繁')) {
        const m = dlg.match(/(\d+)\s*秒/);
        const ok = page.getByRole('button', { name: '确定' });
        if (await ok.count()) await ok.first().click();
        await page.waitForTimeout(((m ? Number(m[1]) : 60) + 10) * 1000);
        continue;
      }
      const now = (await latest()).ID;
      if (now !== before) return true;
      await page.waitForTimeout(8000);
    }
    return false;
  };

  await page.route('**/__exp_list.txt', route => route.fulfill({
    path: require('path').resolve(process.cwd(), '.tmp_exp/list.txt'),
    contentType: 'text/plain; charset=utf-8'
  }));
  const list = (await page.evaluate(() =>
    fetch('/__exp_list.txt', { cache: 'no-store' }).then(r => r.text())))
    .split('\n').map(s => s.trim()).filter(Boolean);

  for (const file of list) {
    let done = null;
    for (let attempt = 0; attempt < 3 && !done; attempt++) {
      if (!await loadInto(file)) { out.push({ file, error: 'editor mismatch' }); break; }
      if (!await doSubmit()) { out.push({ file, error: 'submit failed' }); break; }
      let r = null;
      for (let w = 0; w < 16; w++) {
        await page.waitForTimeout(30000);
        r = await latest();
        if (r.status !== 'Running') break;
      }
      if (r && r.cases === 15 && /[^0.]/.test(r.prec.replace(/[.,]/g, ''))) done = r;
      else await page.waitForTimeout(60000);
    }
    out.push({ file: file.split('/').pop(), result: done });
  }
  return out;
}
