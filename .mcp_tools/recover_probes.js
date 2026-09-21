async (page) => {
  return await page.evaluate(async () => {
    const uid = '6aabe4f6b0477ec41ec44882';
    const pid = '6a9aa054bf41025d6014f3ef';
    const list = await (await fetch(
      `/api/submissions/user/${uid}/problem/${pid}`,
      { credentials: 'include' })).json();
    const found = [];
    for (const item of list) {
      const value = await (await fetch(`/api/submissions/${item._id}`,
        { credentials: 'include' })).json();
      const kernel = value.files?.find(file => file.path === 'kernel.asc')?.content || '';
      const mode = kernel.match(/#define BMMS_PROBE_MODE\s+(\d+)/)?.[1];
      if (mode && mode !== '0') {
        found.push({
          ID: item.ID,
          mode: Number(mode),
          status: value.status,
          precision: value.result?.map(x => Number(x.precision_ratio).toFixed(2)),
          times: value.result?.map(x => x.time),
        });
      }
    }
    return { count: list.length, found };
  });
}
