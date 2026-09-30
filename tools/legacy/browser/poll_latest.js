async (page) => {
  return await page.evaluate(async () => {
    const uid = '6aabe4f6b0477ec41ec44882';
    const pid = '6a9aa054bf41025d6014f3ef';
    const list = await (await fetch(`/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    const out = [];
    for (const s of list.slice(0, 3)) {
      const j = await (await fetch(`/api/submissions/${s._id}`,
        { credentials: 'include' })).json();
      const res = j.result || [];
      out.push({
        ID: s.ID, time: s.create_time, status: j.status, cases: res.length,
        precision: res.map(x => Number(x.precision_ratio).toFixed(2)).join(','),
        seconds: res.map(x => x.time).join(','),
        best: res.map(x => x.best_time).join(','),
        states: [...new Set(res.map(x => x.testcase_status))].join('/'),
      });
    }
    return out;
  });
}
