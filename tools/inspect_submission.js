async (page) => {
  return await page.evaluate(async () => {
    const value = await (await fetch(
      '/api/submissions/6aaf71e3b0477ec41e4f0fcc',
      { credentials: 'include' })).json();
    return {
      status: value.status,
      topKeys: Object.keys(value),
      result: value.result,
    };
  });
}
