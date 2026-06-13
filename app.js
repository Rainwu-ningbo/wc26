let data = window.WORLD_CUP_DATA ?? { matches: [] };
let matches = data.matches ?? [];

const elements = {
  generatedAt: document.querySelector("#generatedAt"),
  matchCount: document.querySelector("#matchCount"),
  strongCount: document.querySelector("#strongCount"),
  avgMargin: document.querySelector("#avgMargin"),
  signalStrip: document.querySelector("#signalStrip"),
  searchInput: document.querySelector("#searchInput"),
  dateFilter: document.querySelector("#dateFilter"),
  groupFilter: document.querySelector("#groupFilter"),
  pickFilter: document.querySelector("#pickFilter"),
  confidenceFilter: document.querySelector("#confidenceFilter"),
  resetFilters: document.querySelector("#resetFilters"),
  resultCount: document.querySelector("#resultCount"),
  compactToggle: document.querySelector("#compactToggle"),
  matchList: document.querySelector("#matchList"),
  emptyState: document.querySelector("#emptyState"),
  dialog: document.querySelector("#matchDialog"),
  dialogClose: document.querySelector("#dialogClose"),
  dialogContent: document.querySelector("#dialogContent"),
};

const percent = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

function teamName(team) {
  return escapeHtml(team.zh || team.name);
}

function scorePick(match, index = 0) {
  return (
    match.prediction.recommendedScores?.[index] ??
    match.prediction.topScores[index] ??
    match.prediction.topScores[0]
  );
}

function populateFilters() {
  const selectedDate = elements.dateFilter.value;
  const selectedGroup = elements.groupFilter.value;
  const dates = [...new Set(matches.map((match) => match.beijingDate))];
  const groups = [...new Set(matches.map((match) => match.group))].sort();

  elements.dateFilter.innerHTML = '<option value="all">全部日期</option>';
  elements.groupFilter.innerHTML = '<option value="all">全部小组</option>';
  dates.forEach((date) => {
    const match = matches.find((item) => item.beijingDate === date);
    elements.dateFilter.insertAdjacentHTML(
      "beforeend",
      `<option value="${date}">${date.slice(5).replace("-", "月")}日 · ${match.weekday}</option>`,
    );
  });
  groups.forEach((group) => {
    elements.groupFilter.insertAdjacentHTML(
      "beforeend",
      `<option value="${group}">${group} 组</option>`,
    );
  });
  elements.dateFilter.value = dates.includes(selectedDate) ? selectedDate : "all";
  elements.groupFilter.value = groups.includes(selectedGroup) ? selectedGroup : "all";
}

function renderSummary() {
  const strong = matches.filter((match) => match.prediction.confidence === "强").length;
  const avgMargin = matches.reduce((sum, match) => sum + match.market.margin, 0) / matches.length;
  const generated = new Date(data.generatedAt);

  elements.generatedAt.textContent = `预测更新：${generated.toLocaleString("zh-CN", {
    timeZone: "Asia/Shanghai",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })}`;
  elements.matchCount.textContent = matches.length;
  elements.strongCount.textContent = strong;
  elements.avgMargin.textContent = percent(avgMargin, 1);

  const signals = [...matches]
    .sort((a, b) => {
      const aMax = Math.max(a.prediction.win, a.prediction.draw, a.prediction.loss);
      const bMax = Math.max(b.prediction.win, b.prediction.draw, b.prediction.loss);
      return bMax - aMax;
    })
    .slice(0, 3);

  elements.signalStrip.innerHTML = signals
    .map((match, index) => {
      const maxProbability = Math.max(match.prediction.win, match.prediction.draw, match.prediction.loss);
      return `
        <article class="signal-card" data-id="${match.id}">
          <span class="signal-rank">0${index + 1}</span>
          <div>
            <strong>${teamName(match.home)} vs ${teamName(match.away)}</strong>
            <span>${match.beijingDate.slice(5)} ${match.beijingTime} · 首选 ${match.prediction.pick} · ${scorePick(match, 0).score} / ${scorePick(match, 1).score}</span>
          </div>
          <b class="signal-probability">${percent(maxProbability)}</b>
        </article>
      `;
    })
    .join("");
}

function probabilityBar(label, value) {
  return `
    <div class="probability">
      <div><span>${label}</span><strong>${percent(value)}</strong></div>
      <span class="bar"><i style="width:${percent(value)}"></i></span>
    </div>
  `;
}

function cardTemplate(match) {
  const mainScore = scorePick(match, 0);
  const backupScore = scorePick(match, 1);
  return `
    <article class="match-card" data-id="${match.id}" tabindex="0">
      <div class="card-top">
        <span class="card-kickoff">${match.beijingTime} · 北京时间</span>
        <span class="group-badge">${match.group} 组</span>
      </div>
      <div class="team-row">
        <div class="team">
          <img src="${escapeHtml(match.home.logo)}" alt="">
          <strong>${teamName(match.home)}</strong>
          <small>${escapeHtml(match.home.abbr)}</small>
        </div>
        <div class="score-pick">
          <div class="score-choice score-choice-main">
            <span>主推</span>
            <strong>${escapeHtml(mainScore.score)}</strong>
            <small>${percent(mainScore.probability, 1)}</small>
          </div>
          <div class="score-choice score-choice-backup">
            <span>备选</span>
            <strong>${escapeHtml(backupScore.score)}</strong>
            <small>${percent(backupScore.probability, 1)}</small>
          </div>
        </div>
        <div class="team">
          <img src="${escapeHtml(match.away.logo)}" alt="">
          <strong>${teamName(match.away)}</strong>
          <small>${escapeHtml(match.away.abbr)}</small>
        </div>
      </div>
      <div class="probability-row">
        ${probabilityBar("胜", match.prediction.win)}
        ${probabilityBar("平", match.prediction.draw)}
        ${probabilityBar("负", match.prediction.loss)}
      </div>
      <div class="odds-row">
        <span class="odd">胜 <strong>${match.market.winOdds.toFixed(2)}</strong></span>
        <span class="odd">平 <strong>${match.market.drawOdds.toFixed(2)}</strong></span>
        <span class="odd">负 <strong>${match.market.lossOdds.toFixed(2)}</strong></span>
      </div>
      <div class="pick-row">
        <span>首选 <b>${match.prediction.pick}</b> · 复式参考 <b>${match.prediction.cover}</b></span>
        <span class="confidence-badge confidence-${match.prediction.confidence}">${match.prediction.confidence}</span>
      </div>
    </article>
  `;
}

function getFilteredMatches() {
  const term = elements.searchInput.value.trim().toLowerCase();
  return matches.filter((match) => {
    const names = `${match.home.zh} ${match.home.name} ${match.away.zh} ${match.away.name}`.toLowerCase();
    return (
      (!term || names.includes(term)) &&
      (elements.dateFilter.value === "all" || match.beijingDate === elements.dateFilter.value) &&
      (elements.groupFilter.value === "all" || match.group === elements.groupFilter.value) &&
      (elements.pickFilter.value === "all" || match.prediction.pick === elements.pickFilter.value) &&
      (elements.confidenceFilter.value === "all" || match.prediction.confidence === elements.confidenceFilter.value)
    );
  });
}

function renderMatches() {
  const filtered = getFilteredMatches();
  let currentDate = "";
  const html = [];

  filtered.forEach((match) => {
    if (currentDate !== match.beijingDate) {
      currentDate = match.beijingDate;
      html.push(`<div class="date-divider">${currentDate} · ${match.weekday}</div>`);
    }
    html.push(cardTemplate(match));
  });

  elements.matchList.innerHTML = html.join("");
  elements.resultCount.textContent = filtered.length;
  elements.emptyState.hidden = filtered.length !== 0;
}

function metricLine(label, value) {
  return `<div class="metric-line"><span>${label}</span><strong>${value}</strong></div>`;
}

function openDialog(match) {
  const mainScore = scorePick(match, 0);
  const backupScore = scorePick(match, 1);
  elements.dialogContent.innerHTML = `
    <div class="dialog-inner">
      <h3 class="dialog-title">${teamName(match.home)} vs ${teamName(match.away)}</h3>
      <p class="dialog-subtitle">${match.beijingDate} ${match.weekday} ${match.beijingTime} · ${match.group} 组 · ${escapeHtml(match.venue)}</p>
      <div class="dialog-hero">
        <div class="dialog-team">
          <img src="${escapeHtml(match.home.logo)}" alt="">
          <strong>${teamName(match.home)}</strong>
        </div>
        <div class="dialog-score">
          <div>
            <span>主推比分</span>
            <strong>${escapeHtml(mainScore.score)}</strong>
            <small>${percent(mainScore.probability, 1)}</small>
          </div>
          <div class="dialog-score-backup">
            <span>备选比分</span>
            <strong>${escapeHtml(backupScore.score)}</strong>
            <small>${percent(backupScore.probability, 1)}</small>
          </div>
        </div>
        <div class="dialog-team">
          <img src="${escapeHtml(match.away.logo)}" alt="">
          <strong>${teamName(match.away)}</strong>
        </div>
      </div>
      <div class="dialog-grid">
        <section class="detail-panel">
          <h4>GPT / GPT 预测</h4>
          ${metricLine("胜 / 平 / 负", `${percent(match.prediction.win)} / ${percent(match.prediction.draw)} / ${percent(match.prediction.loss)}`)}
          ${metricLine("GPT 判断", match.prediction.analysis || "暂无简评")}
          ${metricLine("主要风险", match.prediction.risk || "暂无风险提示")}
          ${metricLine("复式参考", match.prediction.cover)}
        </section>
        <section class="detail-panel">
          <h4>MARKET / 公开市场参考</h4>
          ${metricLine("胜 / 平 / 负", `${match.market.winOdds.toFixed(2)} / ${match.market.drawOdds.toFixed(2)} / ${match.market.lossOdds.toFixed(2)}`)}
          ${metricLine("大小球盘口", `${match.market.totalLine} 球`)}
          ${metricLine("大 / 小赔率", `${match.market.overOdds.toFixed(2)} / ${match.market.underOdds.toFixed(2)}`)}
          ${metricLine("市场水位", percent(match.market.margin, 1))}
          ${metricLine("模型公平赔率", `${match.prediction.fairOdds.win} / ${match.prediction.fairOdds.draw} / ${match.prediction.fairOdds.loss}`)}
        </section>
        <section class="detail-panel">
          <h4>SCORELINE / GPT 比分</h4>
          ${match.prediction.topScores
            .map(
              (item) =>
                `<div class="score-line"><span>${escapeHtml(item.score)}</span><strong>${percent(item.probability, 1)} · 公平赔 ${item.fairOdds}</strong></div>`,
            )
            .join("")}
        </section>
        <section class="detail-panel">
          <h4>READ / 快速判断</h4>
          ${metricLine("首选赛果", match.prediction.pick)}
          ${metricLine("信心等级", match.prediction.confidence)}
          ${metricLine("市场预期进球", `${match.prediction.homeXg} - ${match.prediction.awayXg}`)}
          ${metricLine("市场大于 2.5 球", percent(match.prediction.over25))}
          ${metricLine("主推 / 备选比分", `${mainScore.score} / ${backupScore.score}`)}
          <p class="source-note">GPT 基于 ${escapeHtml(match.market.source)} 公开市场参考线与市场派生指标作出判断。并非中国体彩实时出票赔率。</p>
        </section>
      </div>
    </div>
  `;
  elements.dialog.showModal();
}

function resetFilters() {
  elements.searchInput.value = "";
  elements.dateFilter.value = "all";
  elements.groupFilter.value = "all";
  elements.pickFilter.value = "all";
  elements.confidenceFilter.value = "all";
  renderMatches();
}

[elements.searchInput, elements.dateFilter, elements.groupFilter, elements.pickFilter, elements.confidenceFilter].forEach(
  (element) => element.addEventListener("input", renderMatches),
);

elements.resetFilters.addEventListener("click", resetFilters);
elements.compactToggle.addEventListener("click", () => {
  const compact = elements.matchList.classList.toggle("compact");
  elements.compactToggle.textContent = compact ? "舒展视图" : "紧凑视图";
});
elements.dialogClose.addEventListener("click", () => elements.dialog.close());
elements.dialog.addEventListener("click", (event) => {
  if (event.target === elements.dialog) elements.dialog.close();
});
document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-id]");
  if (!target) return;
  const match = matches.find((item) => item.id === target.dataset.id);
  if (match) openDialog(match);
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const target = event.target.closest(".match-card");
  if (!target) return;
  const match = matches.find((item) => item.id === target.dataset.id);
  if (match) openDialog(match);
});

// Render embedded data immediately, then replace it with live public-market data.
async function bootstrap() {
  populateFilters();
  renderSummary();
  renderMatches();

  const refresh = async () => {
    if (location.protocol.startsWith("http")) {
      try {
        const response = await fetch(`predictions.json?v=${Date.now()}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        data = await response.json();
        matches = data.matches ?? [];
        populateFilters();
        renderSummary();
        renderMatches();
      } catch (error) {
        console.warn("GPT 预测数据读取失败，保留最近数据", error);
      }
    }
  };

  await refresh();
  window.setInterval(refresh, 10 * 60 * 1000);
}

bootstrap();
