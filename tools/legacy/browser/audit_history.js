async (page) => {
  return await page.evaluate(async () => {
    const uid = '6aabe4f6b0477ec41ec44882';
    const pid = '6a9aa054bf41025d6014f3ef';
    const list = await (await fetch(`/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    const rows = (await Promise.all(list.map(async item => {
      const j = await (await fetch(`/api/submissions/${item._id}`,
        { credentials: 'include' })).json();
      const file = (j.files || []).find(x => x.path.endsWith('kernel.asc'));
      const code = file?.content || '';
      const stride = code.match(/#define\s+BMMS_TAIL_STRIDE\s+(\d+)/)?.[1] ?? null;
      if (stride === '1' || code.includes('useGmTailPath')) {
        const bytes = new TextEncoder().encode(code.replace(/\r\n/g, '\n'));
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        const hash = [...new Uint8Array(digest)]
          .map(x => x.toString(16).padStart(2, '0')).join('');
        return { ID: item.ID, id: item._id, status: j.status, stride,
          gm: code.includes('useGmTailPath'), hash,
          result: (j.result || []).map(x => ({ p: x.precision_ratio, t: x.time })) };
      }
      return null;
    }))).filter(Boolean);
    return rows;
  });
}
