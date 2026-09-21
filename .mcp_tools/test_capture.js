async (page) => {
  return await page.evaluate(async () => {
    const clipboard = navigator.clipboard;
    const proto = Object.getPrototypeOf(clipboard);
    const own = Object.getOwnPropertyDescriptor(clipboard, 'writeText');
    const inherited = Object.getOwnPropertyDescriptor(proto, 'writeText');
    window.__capturedCopy = null;
    const original = clipboard.writeText.bind(clipboard);
    Object.defineProperty(clipboard, 'writeText', {
      configurable: true,
      value: async (text) => {
        window.__capturedCopy = text;
        return original(text);
      },
    });
    document.querySelector('button[title="复制当前文件"]').click();
    await new Promise(resolve => setTimeout(resolve, 500));
    return {
      own: Boolean(own),
      inherited: Boolean(inherited),
      captured: window.__capturedCopy?.length ?? -1,
      head: window.__capturedCopy?.slice(0, 80) ?? null,
    };
  });
}
