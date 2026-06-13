#!/usr/bin/env python3
"""Refresh remaining 2026 World Cup group-stage predictions from public market lines."""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from world_cup_predictor import MatchInput, predict_match  # noqa: E402


SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    "fifa.world/scoreboard?dates=20260611-20260628&limit=100"
)
BEIJING = ZoneInfo("Asia/Shanghai")

ZH_NAMES = {
    "Algeria": "阿尔及利亚",
    "Argentina": "阿根廷",
    "Australia": "澳大利亚",
    "Austria": "奥地利",
    "Belgium": "比利时",
    "Bosnia-Herzegovina": "波黑",
    "Brazil": "巴西",
    "Canada": "加拿大",
    "Cape Verde": "佛得角",
    "Cape Verde Islands": "佛得角",
    "Colombia": "哥伦比亚",
    "Congo DR": "民主刚果",
    "Croatia": "克罗地亚",
    "Curaçao": "库拉索",
    "Czechia": "捷克",
    "Ecuador": "厄瓜多尔",
    "Egypt": "埃及",
    "England": "英格兰",
    "France": "法国",
    "Germany": "德国",
    "Ghana": "加纳",
    "Haiti": "海地",
    "Iran": "伊朗",
    "Iraq": "伊拉克",
    "Ivory Coast": "科特迪瓦",
    "Japan": "日本",
    "Jordan": "约旦",
    "Mexico": "墨西哥",
    "Morocco": "摩洛哥",
    "Netherlands": "荷兰",
    "New Zealand": "新西兰",
    "Norway": "挪威",
    "Panama": "巴拿马",
    "Paraguay": "巴拉圭",
    "Portugal": "葡萄牙",
    "Qatar": "卡塔尔",
    "Saudi Arabia": "沙特阿拉伯",
    "Scotland": "苏格兰",
    "Senegal": "塞内加尔",
    "South Africa": "南非",
    "South Korea": "韩国",
    "Spain": "西班牙",
    "Sweden": "瑞典",
    "Switzerland": "瑞士",
    "Tunisia": "突尼斯",
    "Türkiye": "土耳其",
    "United States": "美国",
    "Uruguay": "乌拉圭",
    "Uzbekistan": "乌兹别克斯坦",
}


def american_to_decimal(value: str | int | float) -> float:
    number = float(value)
    if number > 0:
        return 1.0 + number / 100.0
    return 1.0 + 100.0 / abs(number)


def poisson_cdf(max_goals: int, rate: float) -> float:
    probability = math.exp(-rate)
    total = probability
    for goals in range(1, max_goals + 1):
        probability *= rate / goals
        total += probability
    return total


def infer_total_goals(line: float, over_odds: float, under_odds: float) -> float:
    over_implied = 1.0 / over_odds
    under_implied = 1.0 / under_odds
    over_probability = over_implied / (over_implied + under_implied)
    cutoff = math.floor(line)

    low, high = 0.35, 7.0
    for _ in range(60):
        rate = (low + high) / 2.0
        model_over = 1.0 - poisson_cdf(cutoff, rate)
        if model_over < over_probability:
            low = rate
        else:
            high = rate
    return (low + high) / 2.0


def get_close_odds(node: dict, side: str) -> str:
    return node[side]["close"]["odds"]


def load_schedule() -> dict:
    request = urllib.request.Request(SCHEDULE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def confidence_label(probability: float) -> str:
    if probability >= 0.72:
        return "强"
    if probability >= 0.58:
        return "中"
    return "谨慎"


def cover_label(probabilities: tuple[float, float, float]) -> str:
    labels = ("胜", "平", "负")
    ordered = sorted(range(3), key=lambda index: probabilities[index], reverse=True)
    if probabilities[ordered[0]] >= 0.68:
        return labels[ordered[0]]
    return f"{labels[ordered[0]]}/{labels[ordered[1]]}"


def round_probability(value: float) -> float:
    return round(value, 4)


def score_matches_pick(score: str, pick_index: int) -> bool:
    home_goals, away_goals = (int(value) for value in score.split("-", 1))
    if pick_index == 0:
        return home_goals > away_goals
    if pick_index == 1:
        return home_goals == away_goals
    return home_goals < away_goals


def build_match(event: dict) -> dict:
    competition = event["competitions"][0]
    competitors = {item["homeAway"]: item for item in competition["competitors"]}
    home = competitors["home"]["team"]
    away = competitors["away"]["team"]
    market = (competition.get("odds") or [])[0]
    moneyline = market["moneyline"]

    win_odds = american_to_decimal(get_close_odds(moneyline, "home"))
    draw_odds = american_to_decimal(get_close_odds(moneyline, "draw"))
    loss_odds = american_to_decimal(get_close_odds(moneyline, "away"))

    total_market = market.get("total", {})
    over_odds = american_to_decimal(total_market["over"]["close"]["odds"])
    under_odds = american_to_decimal(total_market["under"]["close"]["odds"])
    total_line = float(market.get("overUnder", 2.5))
    total_goals = infer_total_goals(total_line, over_odds, under_odds)

    prediction = predict_match(
        MatchInput(
            match_id=event["id"],
            home_team=home["displayName"],
            away_team=away["displayName"],
            win_odds=win_odds,
            draw_odds=draw_odds,
            loss_odds=loss_odds,
            total_goals_prior=total_goals,
            prior_weight=0.10,
            dc_rho=-0.08,
        ),
        top_n=5,
    )

    kickoff = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    beijing = kickoff.astimezone(BEIJING)
    probabilities = (
        prediction.model_win_prob,
        prediction.model_draw_prob,
        prediction.model_loss_prob,
    )
    labels = ("胜", "平", "负")
    pick_index = max(range(3), key=lambda index: probabilities[index])
    recommended_scores = [
        score for score in prediction.top_scores if score_matches_pick(score.score, pick_index)
    ][:2]
    group_note = competition.get("altGameNote", "")
    group = group_note.rsplit(" ", 1)[-1] if group_note else "?"

    return {
        "id": event["id"],
        "kickoffUtc": kickoff.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "beijingDate": beijing.strftime("%Y-%m-%d"),
        "beijingTime": beijing.strftime("%H:%M"),
        "weekday": "周" + "一二三四五六日"[beijing.weekday()],
        "group": group,
        "venue": competition.get("venue", {}).get("fullName", ""),
        "home": {
            "name": home["displayName"],
            "zh": ZH_NAMES.get(home["displayName"], home["displayName"]),
            "abbr": home.get("abbreviation", ""),
            "logo": home.get("logo", ""),
        },
        "away": {
            "name": away["displayName"],
            "zh": ZH_NAMES.get(away["displayName"], away["displayName"]),
            "abbr": away.get("abbreviation", ""),
            "logo": away.get("logo", ""),
        },
        "market": {
            "source": market.get("provider", {}).get("displayName", "公开市场"),
            "winOdds": round(win_odds, 2),
            "drawOdds": round(draw_odds, 2),
            "lossOdds": round(loss_odds, 2),
            "margin": round_probability(prediction.market_margin),
            "totalLine": total_line,
            "overOdds": round(over_odds, 2),
            "underOdds": round(under_odds, 2),
        },
        "prediction": {
            "win": round_probability(prediction.model_win_prob),
            "draw": round_probability(prediction.model_draw_prob),
            "loss": round_probability(prediction.model_loss_prob),
            "homeXg": round(prediction.home_xg, 2),
            "awayXg": round(prediction.away_xg, 2),
            "totalXg": round(prediction.expected_total_goals, 2),
            "over25": round_probability(prediction.over_2_5_prob),
            "btts": round_probability(prediction.both_teams_score_prob),
            "pick": labels[pick_index],
            "cover": cover_label(probabilities),
            "confidence": confidence_label(probabilities[pick_index]),
            "topScore": prediction.top_scores[0].score,
            "topScores": [
                {
                    "score": score.score,
                    "probability": round_probability(score.probability),
                    "fairOdds": round(score.fair_odds, 2),
                }
                for score in prediction.top_scores
            ],
            "recommendedScores": [
                {
                    "score": score.score,
                    "probability": round_probability(score.probability),
                    "fairOdds": round(score.fair_odds, 2),
                }
                for score in recommended_scores
            ],
            "fairOdds": {
                "win": round(1.0 / prediction.model_win_prob, 2),
                "draw": round(1.0 / prediction.model_draw_prob, 2),
                "loss": round(1.0 / prediction.model_loss_prob, 2),
            },
        },
    }


def main() -> int:
    data = load_schedule()
    events = [
        event
        for event in data.get("events", [])
        if event.get("season", {}).get("slug") == "group-stage"
        and event["status"]["type"].get("state") == "pre"
    ]
    events.sort(key=lambda event: event["date"])
    matches = [build_match(event) for event in events]
    generated_at = datetime.now(BEIJING).isoformat(timespec="seconds")
    payload = {
        "generatedAt": generated_at,
        "timezone": "Asia/Shanghai",
        "sourceUrl": SCHEDULE_URL,
        "matches": matches,
    }

    (ROOT / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "predictions.js").write_text(
        "window.WORLD_CUP_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(matches)} 场预测，更新时间 {generated_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
