async (page) => {
  return await page.evaluate(async () => {
    const uid = '6aabe4f6b0477ec41ec44882';
    const pid = '6a9aa054bf41025d6014f3ef';
    const list = await (await fetch(
      `/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    const item = list.find(entry => entry.ID === 369349);
    const value = await (await fetch(`/api/submissions/${item._id}`,
      { credentials: 'include' })).json();
    const kernel = value.files.find(file => file.path === 'kernel.asc').content;
    const lines = kernel.replace(/\r\n/g, '\n').split('\n');
    const matches = [];
    for (let i = 0; i < lines.length; ++i) {
      if (/Dot|dot|m == 1|TPipe/.test(lines[i])) {
        matches.push({ line: i + 1, text: lines.slice(
          Math.max(0, i - 3), Math.min(lines.length, i + 8)).join('\n') });
      }
    }
    return { id: item._id, matches };
  });
}
