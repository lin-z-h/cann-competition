async (page) => {
  const EDITOR = '.monaco-editor textarea.inputarea';
  const URL = 'https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submit';
  const modes = [6, 16];
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

  // Click submit, absorbing the submission-rate cooldown until a new
  // submission id actually appears.
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

  for (const mode of modes) {
    let done = null;
    for (let attempt = 0; attempt < 3 && !done; attempt++) {
      if (!await loadInto(`D:/cann_competition/.tmp_probes/kernel.p${mode}.asc`)) {
        out.push({ mode, error: 'editor mismatch' });
        break;
      }
      if (!await doSubmit()) { out.push({ mode, error: 'submit failed' }); break; }
      let r = null;
      for (let w = 0; w < 16; w++) {
        await page.waitForTimeout(30000);
        r = await latest();
        if (r.status !== 'Running') break;
      }
      if (r && r.cases === 15 && /[^0.]/.test(r.prec.replace(/[.,]/g, ''))) done = r;
      else await page.waitForTimeout(60000);
    }
    out.push({ mode, result: done });
  }
  return out;
}
