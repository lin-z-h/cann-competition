async (page) => {
  return await page.evaluate(async () => {
    const value = await (await fetch(
      '/api/submissions/6aaf3199b0477ec41e320ccb',
      { credentials: 'include' })).json();
    return value.files?.map(file => ({
      keys: Object.keys(file),
      path: file.path,
      name: file.name,
      contentType: typeof file.content,
      contentLength: file.content?.length,
    }));
  });
}
