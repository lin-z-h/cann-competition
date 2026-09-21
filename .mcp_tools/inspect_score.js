async (page) => {
  return await page.evaluate(async () => {
    const submissionId = '6aaf6f2eb0477ec41e4d6aa6';
    const submission = await (await fetch(
      `/api/submissions/${submissionId}`,
      { credentials: 'include' })).json();
    const ranking = await (await fetch(
      '/api/problems/6a9aa054bf41025d6014f3ef/ranking?current=1&size=300',
      { credentials: 'include' })).json();
    const own = ranking.rows.find(row =>
      row.user_id === '6aabe4f6b0477ec41ec44882');
    return {
      submission: {
        ID: submission.ID,
        status: submission.status,
        score: submission.score,
        total_score: submission.total_score,
        result_score: submission.result_score,
      },
      ranking: own ? {
        rank: own.rank,
        score: own.score,
        submissionID: own.submission?.ID,
      } : { found: false, total: ranking.total, returned: ranking.rows.length },
    };
  });
}
