(() => {
  const SCHEDULE_URL =
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260611-20260628&limit=100";
  const MAX_GOALS = 10;

  const americanToDecimal = (value) => {
    const number = Number(value);
    return number > 0 ? 1 + number / 100 : 1 + 100 / Math.abs(number);
  };

  const poissonCdf = (maxGoals, rate) => {
    let probability = Math.exp(-rate);
    let total = probability;
    for (let goals = 1; goals <= maxGoals; goals += 1) {
      probability *= rate / goals;
      total += probability;
    }
    return total;
  };

  const inferTotalGoals = (line, overOdds, underOdds) => {
    const overImplied = 1 / overOdds;
    const underImplied = 1 / underOdds;
    const overProbability = overImplied / (overImplied + underImplied);
    const cutoff = Math.floor(line);
    let low = 0.35;
    let high = 7;

    for (let iteration = 0; iteration < 60; iteration += 1) {
      const rate = (low + high) / 2;
      const modelOver = 1 - poissonCdf(cutoff, rate);
      if (modelOver < overProbability) low = rate;
      else high = rate;
    }
    return (low + high) / 2;
  };

  const removeOverround = (odds) => {
    const implied = odds.map((value) => 1 / value);
    const book = implied.reduce((sum, value) => sum + value, 0);
    return {
      probabilities: implied.map((value) => value / book),
      margin: book - 1,
    };
  };

  const poissonProbabilities = (rate) => {
    const probabilities = [Math.exp(-rate)];
    for (let goals = 1; goals <= MAX_GOALS; goals += 1) {
      probabilities.push((probabilities[goals - 1] * rate) / goals);
    }
    return probabilities;
  };

  const dixonColesTau = (homeGoals, awayGoals, homeXg, awayXg, rho) => {
    if (homeGoals === 0 && awayGoals === 0) return 1 - homeXg * awayXg * rho;
    if (homeGoals === 0 && awayGoals === 1) return 1 + homeXg * rho;
    if (homeGoals === 1 && awayGoals === 0) return 1 + awayXg * rho;
    if (homeGoals === 1 && awayGoals === 1) return 1 - rho;
    return 1;
  };

  const scoreMatrix = (homeXg, awayXg, rho = -0.08) => {
    const homeProbabilities = poissonProbabilities(homeXg);
    const awayProbabilities = poissonProbabilities(awayXg);
    const matrix = [];
    let total = 0;

    homeProbabilities.forEach((homeProbability, homeGoals) => {
      const row = [];
      awayProbabilities.forEach((awayProbability, awayGoals) => {
        const probability = Math.max(
          homeProbability *
            awayProbability *
            dixonColesTau(homeGoals, awayGoals, homeXg, awayXg, rho),
          0,
        );
        row.push(probability);
        total += probability;
      });
      matrix.push(row);
    });
    return matrix.map((row) => row.map((probability) => probability / total));
  };

  const outcomeProbabilities = (matrix) => {
    const result = [0, 0, 0];
    matrix.forEach((row, homeGoals) => {
      row.forEach((probability, awayGoals) => {
        if (homeGoals > awayGoals) result[0] += probability;
        else if (homeGoals === awayGoals) result[1] += probability;
        else result[2] += probability;
      });
    });
    return result;
  };

  const fitObjective = (homeXg, awayXg, target, totalGoalsPrior, priorWeight, rho) => {
    const predicted = outcomeProbabilities(scoreMatrix(homeXg, awayXg, rho));
    const brierLoss = predicted.reduce(
      (loss, value, index) => loss + (value - target[index]) ** 2,
      0,
    );
    return brierLoss + priorWeight * (homeXg + awayXg - totalGoalsPrior) ** 2;
  };

  const fitExpectedGoals = (target, totalGoalsPrior, priorWeight = 0.1, rho = -0.08) => {
    let bestHome = 1.3;
    let bestAway = 1.3;
    let bestLoss = Number.POSITIVE_INFINITY;

    for (let homeStep = 2; homeStep < 34; homeStep += 1) {
      const homeXg = homeStep * 0.12;
      for (let awayStep = 2; awayStep < 34; awayStep += 1) {
        const awayXg = awayStep * 0.12;
        const loss = fitObjective(homeXg, awayXg, target, totalGoalsPrior, priorWeight, rho);
        if (loss < bestLoss) {
          [bestHome, bestAway, bestLoss] = [homeXg, awayXg, loss];
        }
      }
    }

    let step = 0.08;
    while (step >= 0.0025) {
      let improved = false;
      [-step, 0, step].forEach((homeDelta) => {
        [-step, 0, step].forEach((awayDelta) => {
          const homeXg = Math.min(Math.max(bestHome + homeDelta, 0.05), 5);
          const awayXg = Math.min(Math.max(bestAway + awayDelta, 0.05), 5);
          const loss = fitObjective(homeXg, awayXg, target, totalGoalsPrior, priorWeight, rho);
          if (loss + 1e-12 < bestLoss) {
            [bestHome, bestAway, bestLoss] = [homeXg, awayXg, loss];
            improved = true;
          }
        });
      });
      if (!improved) step /= 2;
    }
    return [bestHome, bestAway];
  };

  const scoreMatchesPick = (score, pickIndex) => {
    const [homeGoals, awayGoals] = score.split("-").map(Number);
    if (pickIndex === 0) return homeGoals > awayGoals;
    if (pickIndex === 1) return homeGoals === awayGoals;
    return homeGoals < awayGoals;
  };

  const predictMarket = (winOdds, drawOdds, lossOdds, totalGoals) => {
    const { probabilities: marketProbabilities, margin } = removeOverround([
      winOdds,
      drawOdds,
      lossOdds,
    ]);
    const [homeXg, awayXg] = fitExpectedGoals(marketProbabilities, totalGoals);
    const matrix = scoreMatrix(homeXg, awayXg);
    const probabilities = outcomeProbabilities(matrix);
    const scores = [];
    let over25 = 0;
    let btts = 0;

    matrix.forEach((row, homeGoals) => {
      row.forEach((probability, awayGoals) => {
        if (homeGoals + awayGoals >= 3) over25 += probability;
        if (homeGoals >= 1 && awayGoals >= 1) btts += probability;
        scores.push({
          score: `${homeGoals}-${awayGoals}`,
          probability,
          fairOdds: 1 / probability,
        });
      });
    });
    scores.sort((a, b) => b.probability - a.probability);
    const pickIndex = probabilities.indexOf(Math.max(...probabilities));
    return {
      margin,
      probabilities,
      homeXg,
      awayXg,
      over25,
      btts,
      scores,
      pickIndex,
      recommendedScores: scores.filter((score) => scoreMatchesPick(score.score, pickIndex)).slice(0, 2),
    };
  };

  const closeOdds = (node, side) => node[side].close.odds;
  const rounded = (value, digits = 4) => Number(value.toFixed(digits));
  const translatedName = (team, names) => names.get(team.displayName) ?? team.displayName;
  const shiftedIso = (date) => new Date(date.getTime() + 8 * 60 * 60 * 1000).toISOString();

  const buildMatch = (event, names) => {
    const competition = event.competitions[0];
    const competitors = Object.fromEntries(
      competition.competitors.map((item) => [item.homeAway, item]),
    );
    const home = competitors.home.team;
    const away = competitors.away.team;
    const market = competition.odds[0];
    const moneyline = market.moneyline;
    const winOdds = americanToDecimal(closeOdds(moneyline, "home"));
    const drawOdds = americanToDecimal(closeOdds(moneyline, "draw"));
    const lossOdds = americanToDecimal(closeOdds(moneyline, "away"));
    const overOdds = americanToDecimal(market.total.over.close.odds);
    const underOdds = americanToDecimal(market.total.under.close.odds);
    const totalLine = Number(market.overUnder ?? 2.5);
    const totalGoals = inferTotalGoals(totalLine, overOdds, underOdds);
    const prediction = predictMarket(winOdds, drawOdds, lossOdds, totalGoals);
    const labels = ["胜", "平", "负"];
    const strongest = Math.max(...prediction.probabilities);
    const ordered = [0, 1, 2].sort(
      (left, right) => prediction.probabilities[right] - prediction.probabilities[left],
    );
    const kickoff = new Date(event.date);
    const beijingIso = shiftedIso(kickoff);
    const group = competition.altGameNote?.split(" ").at(-1) ?? "?";
    const mappedScores = (scores) =>
      scores.map((score) => ({
        score: score.score,
        probability: rounded(score.probability),
        fairOdds: rounded(score.fairOdds, 2),
      }));

    return {
      id: event.id,
      kickoffUtc: kickoff.toISOString(),
      beijingDate: beijingIso.slice(0, 10),
      beijingTime: beijingIso.slice(11, 16),
      weekday: new Intl.DateTimeFormat("zh-CN", {
        timeZone: "Asia/Shanghai",
        weekday: "short",
      }).format(kickoff),
      group,
      venue: competition.venue?.fullName ?? "",
      home: {
        name: home.displayName,
        zh: translatedName(home, names),
        abbr: home.abbreviation ?? "",
        logo: home.logo ?? "",
      },
      away: {
        name: away.displayName,
        zh: translatedName(away, names),
        abbr: away.abbreviation ?? "",
        logo: away.logo ?? "",
      },
      market: {
        source: market.provider?.displayName ?? "公开市场",
        winOdds: rounded(winOdds, 2),
        drawOdds: rounded(drawOdds, 2),
        lossOdds: rounded(lossOdds, 2),
        margin: rounded(prediction.margin),
        totalLine,
        overOdds: rounded(overOdds, 2),
        underOdds: rounded(underOdds, 2),
      },
      prediction: {
        win: rounded(prediction.probabilities[0]),
        draw: rounded(prediction.probabilities[1]),
        loss: rounded(prediction.probabilities[2]),
        homeXg: rounded(prediction.homeXg, 2),
        awayXg: rounded(prediction.awayXg, 2),
        totalXg: rounded(prediction.homeXg + prediction.awayXg, 2),
        over25: rounded(prediction.over25),
        btts: rounded(prediction.btts),
        pick: labels[prediction.pickIndex],
        cover:
          strongest >= 0.68
            ? labels[ordered[0]]
            : `${labels[ordered[0]]}/${labels[ordered[1]]}`,
        confidence: strongest >= 0.72 ? "强" : strongest >= 0.58 ? "中" : "谨慎",
        topScore: prediction.scores[0].score,
        topScores: mappedScores(prediction.scores.slice(0, 5)),
        recommendedScores: mappedScores(prediction.recommendedScores),
        fairOdds: {
          win: rounded(1 / prediction.probabilities[0], 2),
          draw: rounded(1 / prediction.probabilities[1], 2),
          loss: rounded(1 / prediction.probabilities[2], 2),
        },
      },
    };
  };

  window.refreshWorldCupData = async (fallback) => {
    const response = await fetch(`${SCHEDULE_URL}&v=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`赛程接口返回 HTTP ${response.status}`);
    const schedule = await response.json();
    const names = new Map();
    fallback.matches.forEach((match) => {
      names.set(match.home.name, match.home.zh);
      names.set(match.away.name, match.away.zh);
    });

    const events = schedule.events
      .filter(
        (event) =>
          event.season?.slug === "group-stage" &&
          event.status?.type?.state === "pre" &&
          event.competitions?.[0]?.odds?.[0]?.moneyline,
      )
      .sort((left, right) => left.date.localeCompare(right.date));

    return {
      generatedAt: shiftedIso(new Date()).replace("Z", "+08:00"),
      timezone: "Asia/Shanghai",
      sourceUrl: SCHEDULE_URL,
      matches: events.map((event) => buildMatch(event, names)),
    };
  };
})();
