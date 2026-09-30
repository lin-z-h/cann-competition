async (page) => {
  const id = '6aaf800db0477ec41e572f93';
  const data = await page.evaluate(async submissionId => {
    const response = await fetch(`/api/submissions/${submissionId}`, {
      credentials: 'include',
    });
    return response.json();
  }, id);
  const summarize = value => {
    if (typeof value === 'string') {
      return value.length > 12000 ? value.slice(0, 12000) : value;
    }
    return value;
  };
  const out = {};
  for (const [key, value] of Object.entries(data)) {
    if (key === 'code') {
      out[key] = `[omitted ${JSON.stringify(value).length} chars]`;
    } else if (key === 'files') {
      let files = value;
      if (typeof files === 'string') {
        try { files = JSON.parse(files); } catch (_) {}
      }
      out[key] = Array.isArray(files) ? files.map(file => ({
        name: file.name || file.filename || file.path,
        keys: Object.keys(file),
        contentLength: typeof file.content === 'string' ? file.content.length : undefined,
        message: file.message || file.error || file.stderr || file.stdout,
      })) : { type: typeof files, preview: String(files).slice(0, 2000) };
    } else {
      out[key] = summarize(value);
    }
  }
  return out;
}
