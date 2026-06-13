#!/usr/bin/env python3
"""World Cup 1X2 and correct-score predictor driven by lottery odds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MAX_GOALS = 10


@dataclass(frozen=True)
class MatchInput:
    match_id: str
    home_team: str
    away_team: str
    win_odds: float
    draw_odds: float
    loss_odds: float
    total_goals_prior: float = 2.6
    prior_weight: float = 0.015
    dc_rho: float = -0.08


@dataclass(frozen=True)
class ScorePrediction:
    score: str
    probability: float
    fair_odds: float


@dataclass(frozen=True)
class MatchPrediction:
    match_id: str
    home_team: str
    away_team: str
    market_margin: float
    market_win_prob: float
    market_draw_prob: float
    market_loss_prob: float
    model_win_prob: float
    model_draw_prob: float
    model_loss_prob: float
    home_xg: float
    away_xg: float
    expected_total_goals: float
    over_2_5_prob: float
    both_teams_score_prob: float
    most_likely_result: str
    top_scores: list[ScorePrediction]


def _validate_odds(odds: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(x) for x in odds)
    if len(values) != 3 or any(not math.isfinite(x) or x <= 1.0 for x in values):
        raise ValueError("胜、平、负三项欧赔都必须是大于 1 的有限数字")
    return values  # type: ignore[return-value]


def remove_overround(
    win_odds: float, draw_odds: float, loss_odds: float
) -> tuple[tuple[float, float, float], float]:
    """Remove bookmaker margin by normalizing inverse decimal odds."""
    odds = _validate_odds((win_odds, draw_odds, loss_odds))
    implied = tuple(1.0 / x for x in odds)
    book = sum(implied)
    probabilities = tuple(x / book for x in implied)
    return probabilities, book - 1.0  # type: ignore[return-value]


def _poisson_probabilities(rate: float, max_goals: int = MAX_GOALS) -> list[float]:
    probabilities = [math.exp(-rate)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * rate / goals)
    return probabilities


def _dixon_coles_tau(home_goals: int, away_goals: int, home_xg: float, away_xg: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_xg * away_xg * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_xg * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_xg * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    home_xg: float, away_xg: float, rho: float = -0.08, max_goals: int = MAX_GOALS
) -> list[list[float]]:
    """Build a normalized correct-score probability matrix."""
    home_probs = _poisson_probabilities(home_xg, max_goals)
    away_probs = _poisson_probabilities(away_xg, max_goals)
    matrix: list[list[float]] = []
    total = 0.0

    for home_goals, home_prob in enumerate(home_probs):
        row = []
        for away_goals, away_prob in enumerate(away_probs):
            probability = home_prob * away_prob * _dixon_coles_tau(
                home_goals, away_goals, home_xg, away_xg, rho
            )
            probability = max(probability, 0.0)
            row.append(probability)
            total += probability
        matrix.append(row)

    return [[probability / total for probability in row] for row in matrix]


def outcome_probabilities(matrix: list[list[float]]) -> tuple[float, float, float]:
    win = draw = loss = 0.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if home_goals > away_goals:
                win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                loss += probability
    return win, draw, loss


def _fit_objective(
    home_xg: float,
    away_xg: float,
    target: tuple[float, float, float],
    total_goals_prior: float,
    prior_weight: float,
    rho: float,
) -> float:
    predicted = outcome_probabilities(score_matrix(home_xg, away_xg, rho))
    brier_loss = sum((actual - expected) ** 2 for actual, expected in zip(predicted, target))
    prior_loss = prior_weight * (home_xg + away_xg - total_goals_prior) ** 2
    return brier_loss + prior_loss


def fit_expected_goals(
    target: tuple[float, float, float],
    total_goals_prior: float = 2.6,
    prior_weight: float = 0.015,
    rho: float = -0.08,
) -> tuple[float, float]:
    """Infer expected goals that reproduce the de-vigged 1X2 market."""
    if not 0.5 <= total_goals_prior <= 7.0:
        raise ValueError("total_goals_prior 必须在 0.5 到 7.0 之间")
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight 必须在 0 到 1 之间")
    if not -0.2 <= rho <= 0.2:
        raise ValueError("dc_rho 必须在 -0.2 到 0.2 之间")

    best_home = best_away = 1.3
    best_loss = float("inf")

    # A coarse global search avoids poor local minima and remains fast for CSV batches.
    for home_step in range(2, 34):
        home_xg = home_step * 0.12
        for away_step in range(2, 34):
            away_xg = away_step * 0.12
            loss = _fit_objective(
                home_xg, away_xg, target, total_goals_prior, prior_weight, rho
            )
            if loss < best_loss:
                best_home, best_away, best_loss = home_xg, away_xg, loss

    step = 0.08
    while step >= 0.0025:
        improved = False
        for home_delta in (-step, 0.0, step):
            for away_delta in (-step, 0.0, step):
                home_xg = min(max(best_home + home_delta, 0.05), 5.0)
                away_xg = min(max(best_away + away_delta, 0.05), 5.0)
                loss = _fit_objective(
                    home_xg, away_xg, target, total_goals_prior, prior_weight, rho
                )
                if loss + 1e-12 < best_loss:
                    best_home, best_away, best_loss = home_xg, away_xg, loss
                    improved = True
        if not improved:
            step /= 2.0

    return best_home, best_away


def predict_match(match: MatchInput, top_n: int = 8) -> MatchPrediction:
    market_probs, margin = remove_overround(
        match.win_odds, match.draw_odds, match.loss_odds
    )
    home_xg, away_xg = fit_expected_goals(
        market_probs, match.total_goals_prior, match.prior_weight, match.dc_rho
    )
    matrix = score_matrix(home_xg, away_xg, match.dc_rho)
    model_probs = outcome_probabilities(matrix)

    scores = [
        ScorePrediction(
            score=f"{home_goals}-{away_goals}",
            probability=probability,
            fair_odds=1.0 / probability,
        )
        for home_goals, row in enumerate(matrix)
        for away_goals, probability in enumerate(row)
    ]
    scores.sort(key=lambda item: item.probability, reverse=True)

    over_2_5 = sum(
        probability
        for home_goals, row in enumerate(matrix)
        for away_goals, probability in enumerate(row)
        if home_goals + away_goals >= 3
    )
    both_teams_score = sum(
        probability
        for home_goals, row in enumerate(matrix)
        for away_goals, probability in enumerate(row)
        if home_goals >= 1 and away_goals >= 1
    )
    result_labels = ("胜", "平", "负")
    likely_result = result_labels[max(range(3), key=lambda index: model_probs[index])]

    return MatchPrediction(
        match_id=match.match_id,
        home_team=match.home_team,
        away_team=match.away_team,
        market_margin=margin,
        market_win_prob=market_probs[0],
        market_draw_prob=market_probs[1],
        market_loss_prob=market_probs[2],
        model_win_prob=model_probs[0],
        model_draw_prob=model_probs[1],
        model_loss_prob=model_probs[2],
        home_xg=home_xg,
        away_xg=away_xg,
        expected_total_goals=home_xg + away_xg,
        over_2_5_prob=over_2_5,
        both_teams_score_prob=both_teams_score,
        most_likely_result=likely_result,
        top_scores=scores[:top_n],
    )


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_prediction(prediction: MatchPrediction) -> None:
    print(f"\n{prediction.match_id}  {prediction.home_team} vs {prediction.away_team}")
    print(f"体彩去水概率  胜 {_format_percent(prediction.market_win_prob)}  "
          f"平 {_format_percent(prediction.market_draw_prob)}  "
          f"负 {_format_percent(prediction.market_loss_prob)}  "
          f"水位 {_format_percent(prediction.market_margin)}")
    print(f"比分模型概率  胜 {_format_percent(prediction.model_win_prob)}  "
          f"平 {_format_percent(prediction.model_draw_prob)}  "
          f"负 {_format_percent(prediction.model_loss_prob)}")
    print(f"预期进球      {prediction.home_xg:.2f} - {prediction.away_xg:.2f}  "
          f"总进球 {prediction.expected_total_goals:.2f}")
    print(f"大于2.5球 {_format_percent(prediction.over_2_5_prob)}  "
          f"双方进球 {_format_percent(prediction.both_teams_score_prob)}  "
          f"首选赛果 {prediction.most_likely_result}")
    print("最可能比分    " + "  ".join(
        f"{item.score} {_format_percent(item.probability)}"
        for item in prediction.top_scores
    ))


def _float_from_row(row: dict[str, str], field: str, default: float) -> float:
    value = row.get(field, "").strip()
    return float(value) if value else default


def read_matches(path: Path) -> list[MatchInput]:
    matches = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"match_id", "home_team", "away_team", "win_odds", "draw_odds", "loss_odds"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少字段: {', '.join(sorted(missing))}")
        for row in reader:
            matches.append(
                MatchInput(
                    match_id=row["match_id"].strip(),
                    home_team=row["home_team"].strip(),
                    away_team=row["away_team"].strip(),
                    win_odds=float(row["win_odds"]),
                    draw_odds=float(row["draw_odds"]),
                    loss_odds=float(row["loss_odds"]),
                    total_goals_prior=_float_from_row(row, "total_goals_prior", 2.6),
                    prior_weight=_float_from_row(row, "prior_weight", 0.015),
                    dc_rho=_float_from_row(row, "dc_rho", -0.08),
                )
            )
    return matches


def write_predictions(path: Path, predictions: list[MatchPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "match_id", "home_team", "away_team", "market_margin",
        "market_win_prob", "market_draw_prob", "market_loss_prob",
        "model_win_prob", "model_draw_prob", "model_loss_prob",
        "home_xg", "away_xg", "expected_total_goals", "over_2_5_prob",
        "both_teams_score_prob", "most_likely_result", "top_scores",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            row = asdict(prediction)
            row["top_scores"] = json.dumps(
                [asdict(item) for item in prediction.top_scores], ensure_ascii=False
            )
            for field, value in list(row.items()):
                if isinstance(value, float):
                    row[field] = round(value, 6)
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于体彩胜平负欧赔反推世界杯比分概率")
    parser.add_argument("--input", type=Path, help="批量输入 CSV")
    parser.add_argument("--output", type=Path, help="批量结果 CSV")
    parser.add_argument("--home", help="单场：主队/第一顺位球队")
    parser.add_argument("--away", help="单场：客队/第二顺位球队")
    parser.add_argument("--odds", nargs=3, type=float, metavar=("WIN", "DRAW", "LOSS"), help="单场胜平负欧赔")
    parser.add_argument("--total-goals-prior", type=float, default=2.6, help="总进球先验，默认 2.6")
    parser.add_argument("--prior-weight", type=float, default=0.015, help="总进球先验权重，默认 0.015")
    parser.add_argument("--rho", type=float, default=-0.08, help="Dixon-Coles 低比分修正，默认 -0.08")
    parser.add_argument("--top", type=int, default=8, help="输出比分数量，默认 8")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.input:
            matches = read_matches(args.input)
        elif args.home and args.away and args.odds:
            matches = [
                MatchInput(
                    match_id="single-match",
                    home_team=args.home,
                    away_team=args.away,
                    win_odds=args.odds[0],
                    draw_odds=args.odds[1],
                    loss_odds=args.odds[2],
                    total_goals_prior=args.total_goals_prior,
                    prior_weight=args.prior_weight,
                    dc_rho=args.rho,
                )
            ]
        else:
            raise ValueError("请提供 --input，或同时提供 --home、--away、--odds")

        predictions = [predict_match(match, max(1, args.top)) for match in matches]
        for prediction in predictions:
            print_prediction(prediction)
        if args.output:
            write_predictions(args.output, predictions)
            print(f"\n已写入: {args.output}")
        return 0
    except (OSError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
