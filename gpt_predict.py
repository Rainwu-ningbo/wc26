#!/usr/bin/env python3
"""Overlay market-derived match data with GPT football predictions."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = ROOT / "predictions.json"
MODEL_ENDPOINT = "https://models.github.ai/inference/chat/completions"
DEFAULT_MODEL = "openai/gpt-4.1"
BEIJING = ZoneInfo("Asia/Shanghai")
SCORE_PATTERN = re.compile(r"^\d{1,2}-\d{1,2}$")
PICK_LABELS = ("胜", "平", "负")


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("缺少可调用 GitHub Models 的 GITHUB_TOKEN") from error


def prompt_matches(matches: list[dict]) -> list[dict]:
    compact = []
    for match in matches:
        prediction = match["prediction"]
        compact.append(
            {
                "id": match["id"],
                "kickoff_beijing": f"{match['beijingDate']} {match['beijingTime']}",
                "group": match["group"],
                "home": match["home"]["zh"],
                "away": match["away"]["zh"],
                "market_decimal_odds": [
                    match["market"]["winOdds"],
                    match["market"]["drawOdds"],
                    match["market"]["lossOdds"],
                ],
                "market_total_line": match["market"]["totalLine"],
                "market_over_under_odds": [
                    match["market"]["overOdds"],
                    match["market"]["underOdds"],
                ],
                "market_de_vig_probabilities": [
                    round(prediction["win"] * 100, 1),
                    round(prediction["draw"] * 100, 1),
                    round(prediction["loss"] * 100, 1),
                ],
                "market_expected_goals": [
                    prediction["homeXg"],
                    prediction["awayXg"],
                ],
                "market_score_candidates": [
                    item["score"] for item in prediction["topScores"][:5]
                ],
            }
        )
    return compact


def build_request(matches: list[dict], model: str) -> dict:
    system = (
        "你是谨慎、校准良好的世界杯赛前预测分析师。"
        "你必须主要依据用户提供的最新市场赔率、去水概率、大小球和预期进球作出判断，"
        "再结合你对国家队实力、风格和世界杯比赛波动性的常识进行有限调整。"
        "不要编造伤停、首发或新闻。不要因为热门球队名称而过度自信。"
        "为每场比赛给出胜平负概率、与首选赛果一致的两个比分、简短理由和主要风险。"
        "比分概率是该精确比分发生的概率，通常不应过高。只返回 JSON。"
    )
    user = {
        "task": "预测以下全部尚未开球的世界杯小组赛",
        "output_contract": {
            "predictions": [
                {
                    "id": "比赛 id，必须原样返回",
                    "win": "主队胜概率，0到100",
                    "draw": "平局概率，0到100",
                    "loss": "客队胜概率，0到100；三项合计100",
                    "pick": "只能是 胜、平、负",
                    "main_score": "与 pick 一致的主推比分，如 1-0",
                    "backup_score": "与 pick 一致且不同于主推的备选比分",
                    "main_score_probability": "主推精确比分概率，0到100",
                    "backup_score_probability": "备选精确比分概率，0到100",
                    "confidence": "只能是 强、中、谨慎",
                    "analysis": "不超过45个中文字的判断理由",
                    "risk": "不超过35个中文字的主要风险",
                }
            ]
        },
        "rules": [
            "不得遗漏比赛，不得增加比赛",
            "胜平负概率必须合计100",
            "两个比分必须与 pick 对应的赛果一致",
            "赔率市场是主要证据，GPT 调整幅度应克制",
        ],
        "matches": prompt_matches(matches),
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 20000,
    }


def call_gpt(matches: list[dict], token: str, model: str) -> dict:
    request = urllib.request.Request(
        MODEL_ENDPOINT,
        data=json.dumps(build_request(matches, model), ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.load(response)
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def normalized_probabilities(item: dict) -> tuple[float, float, float]:
    values = [max(0.0, float(item.get(key, 0.0))) for key in ("win", "draw", "loss")]
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("GPT 返回了无效的胜平负概率")
    return tuple(value / total for value in values)  # type: ignore[return-value]


def score_outcome(score: str) -> str:
    home_goals, away_goals = (int(value) for value in score.split("-", 1))
    if home_goals > away_goals:
        return "胜"
    if home_goals == away_goals:
        return "平"
    return "负"


def valid_score(score: object, pick: str) -> bool:
    return (
        isinstance(score, str)
        and SCORE_PATTERN.fullmatch(score) is not None
        and score_outcome(score) == pick
    )


def fallback_scores(prediction: dict, pick: str) -> list[dict]:
    scores = [
        item for item in prediction.get("recommendedScores", []) if score_outcome(item["score"]) == pick
    ]
    if len(scores) >= 2:
        return scores[:2]
    scores.extend(
        item
        for item in prediction.get("topScores", [])
        if score_outcome(item["score"]) == pick and item["score"] not in {score["score"] for score in scores}
    )
    return scores[:2]


def score_item(score: str, probability_percent: object) -> dict:
    probability = max(0.005, min(float(probability_percent) / 100.0, 0.45))
    return {
        "score": score,
        "probability": round(probability, 4),
        "fairOdds": round(1.0 / probability, 2),
    }


def cover_label(probabilities: tuple[float, float, float]) -> str:
    ordered = sorted(range(3), key=lambda index: probabilities[index], reverse=True)
    if probabilities[ordered[0]] >= 0.68:
        return PICK_LABELS[ordered[0]]
    return f"{PICK_LABELS[ordered[0]]}/{PICK_LABELS[ordered[1]]}"


def apply_gpt(payload: dict, gpt_output: dict, model: str) -> int:
    raw_items = gpt_output.get("predictions")
    if not isinstance(raw_items, list):
        raise ValueError("GPT 输出缺少 predictions 数组")
    by_id = {str(item.get("id")): item for item in raw_items if isinstance(item, dict)}
    updated = 0

    for match in payload["matches"]:
        item = by_id.get(str(match["id"]))
        if not item:
            continue
        try:
            probabilities = normalized_probabilities(item)
            pick_index = max(range(3), key=lambda index: probabilities[index])
            pick = PICK_LABELS[pick_index]
            main_score = item.get("main_score")
            backup_score = item.get("backup_score")

            if (
                not valid_score(main_score, pick)
                or not valid_score(backup_score, pick)
                or main_score == backup_score
            ):
                fallback = fallback_scores(match["prediction"], pick)
                if len(fallback) < 2:
                    raise ValueError("没有可用的比分备选")
                scores = fallback
            else:
                scores = [
                    score_item(main_score, item.get("main_score_probability", 8)),
                    score_item(backup_score, item.get("backup_score_probability", 6)),
                ]

            prediction = match["prediction"]
            prediction.update(
                {
                    "win": round(probabilities[0], 4),
                    "draw": round(probabilities[1], 4),
                    "loss": round(probabilities[2], 4),
                    "pick": pick,
                    "cover": cover_label(probabilities),
                    "confidence": str(item.get("confidence", "谨慎"))
                    if item.get("confidence") in {"强", "中", "谨慎"}
                    else "谨慎",
                    "topScore": scores[0]["score"],
                    "topScores": scores,
                    "recommendedScores": scores,
                    "fairOdds": {
                        "win": round(1.0 / probabilities[0], 2),
                        "draw": round(1.0 / probabilities[1], 2),
                        "loss": round(1.0 / probabilities[2], 2),
                    },
                    "analysis": str(item.get("analysis", ""))[:90],
                    "risk": str(item.get("risk", ""))[:70],
                    "generatedBy": "GPT",
                    "modelName": model,
                }
            )
            updated += 1
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            print(f"跳过无效 GPT 预测 {match['id']}: {error}", file=sys.stderr)

    payload["generatedAt"] = datetime.now(BEIJING).isoformat(timespec="seconds")
    payload["predictionEngine"] = f"OpenAI {model.removeprefix('openai/')} via GitHub Models"
    payload["gptPredictedMatches"] = updated
    return updated


def write_payload(payload: dict) -> None:
    PREDICTIONS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "predictions.js").write_text(
        "window.WORLD_CUP_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


def main() -> int:
    model = os.environ.get("GPT_MODEL", DEFAULT_MODEL)
    payload = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    if not matches:
        print("没有待预测比赛")
        return 0

    try:
        output = call_gpt(matches, get_token(), model)
        updated = apply_gpt(payload, output, model)
        if updated < max(1, len(matches) // 2):
            raise ValueError(f"GPT 仅返回 {updated}/{len(matches)} 场有效预测")
        write_payload(payload)
        print(f"GPT 已预测 {updated}/{len(matches)} 场，模型 {model}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(f"GPT 预测失败，保留市场模型结果: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
