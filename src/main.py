"""SoccerSim 策略入口 —— 比赛策略主逻辑都在这里,改打法就改这个文件。

三层架构:
1. main.py(策略层):比赛策略。play() 按 Phase 状态机分派到 _act_*;各 _act_* 选出
   attacker(离球最近)并直接调 player 动作。
2. player.py(动作层):Player 控制 handle + 高层动作(attack / guard / support);
   想加拐棍/技术动作直接改它。
3. motion.py(走位层):走位/避障工具(walk_to / face_to / ensure_ready)。

utils/:几何工具(dist / angle_to / clamp ...)。
framework/:平台管线,用户不改。

改打法主要改本文件:Phase 状态机、各 _act_* 行为、站位公式。

数据访问路径:
- context.game: 裁判机状态(包含 phase 等)
- context.strategy: 策略跨帧状态(可修改)
- context.prev_phase: 上一帧的比赛阶段(检测阶段变化)
"""

from __future__ import annotations

import logging

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import Action, Phase, SetType, Context
from .param import *
from .player import Player
from .utils import dist


_log = logging.getLogger(__name__)


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent。"""

    player_class = Player

    @staticmethod
    def play(context: Context, players: list[Player]) -> None:
        """每帧策略入口。根据比赛阶段分派球员动作。"""

        _analyze_and_draw(context, players)

        active_players = [p for p in players if p.check_ready()]
        if not active_players:
            _draw_teammate_markers(players)
            return
        
        phase = context.game.phase

        if phase == Phase.NORMAL:
            _act_normal(context, active_players)
        elif phase == Phase.OUR_KICKOFF:
            _act_our_kickoff(context, active_players)
        elif phase == Phase.OPP_KICKOFF:
            _act_opp_kickoff(context, active_players)
        elif phase == Phase.OUR_SET_PLAY:
            _act_our_set_play(context, active_players)
        elif phase == Phase.OPP_SET_PLAY:
            _act_opp_set_play(context, active_players)
        elif phase == Phase.READY:
            _act_ready(context, active_players)
        elif phase == Phase.STOPPED:
            for p in active_players:
                p.stop()

        _draw_teammate_markers(players)


def _select_closest_attacker(context: Context, players: list[Player]) -> Player:
    """选择到球距离最近的球员作为进攻者(含粘性选择和摔倒惩罚)。
    
    同时更新 context.strategy.normal_attacker_id 实现粘性选择。
    """

    ball = context.ball
    if ball is None:
        chosen = players[0]
        context.strategy.normal_attacker_id = chosen.id
        return chosen

    def dist_to_ball(p: Player) -> float:
        return dist(p.pose.x, p.pose.y, ball.x, ball.y) + (FALLEN_COST if p.is_fallen else 0.0)

    preferred_id = context.strategy.normal_attacker_id
    ranked = [(p, dist_to_ball(p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if preferred is not None and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M:
        chosen = preferred[0]
    else:
        chosen = best
    
    context.strategy.normal_attacker_id = chosen.id
    return chosen


def _act_normal(context: Context, players: list[Player]) -> None:
    """NORMAL阶段:距球最近者attack,剩下人里离己方门最近者guard,其余support。"""

    attacker = _select_closest_attacker(context, players)
    attacker.attack()

    rest = [p for p in players if p is not attacker]
    if rest:
        gx, gy = context.field.own_goal
        guard = min(rest, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
        guard.guard()
        rest = [p for p in rest if p is not guard]

    for p in rest:
        p.support()


def _act_our_kickoff(context: Context, players: list[Player]) -> None:
    """OUR_KICKOFF阶段:第一个人开球,其余人站到开球站位。"""

    context.strategy.normal_attacker_id = None

    attacker = players[0]
    attacker.take_kickoff()

    rest = players[1:]
    Player.walk_to_slots(rest, OUR_KICKOFF_SLOTS, Action.KICKOFF)


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """OPP_KICKOFF阶段:一人守门,其余人站到开球站位等待。"""

    context.strategy.normal_attacker_id = None
    gx, gy = context.field.own_goal
    guard = min(players, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
    guard.guard()

    rest = [p for p in players if p is not guard]
    Player.walk_to_slots(rest, OPP_KICKOFF_SLOTS, Action.OPP_KICKOFF_READY, face=0.0)


def _act_our_set_play(context: Context, players: list[Player]) -> None:
    """OUR_SET_PLAY阶段:按定位球类型分派动作。
    可以让发球队员使用take_kickoff方法，其他队员使用walk_to_slots方法。
    """
    context.strategy.normal_attacker_id = None
    set_type = context.game.set_type
    if set_type == SetType.THROW_IN:
        _act_normal(context, players)
        return
    if set_type == SetType.CORNER_KICK:
        _act_normal(context, players)
        return
    if set_type == SetType.GOAL_KICK:
        _act_normal(context, players)
        return
    _act_normal(context, players)


def _act_opp_set_play(context: Context, players: list[Player]) -> None:
    """OPP_SET_PLAY阶段:对方定位球,全部静止等待。"""
    for p in players:
        p.stop()


def _act_ready(context: Context, players: list[Player]) -> None:
    """READY阶段:各自走到准备位置。站位坐标由 param.py 配置。"""
    context.strategy.normal_attacker_id = None
    is_our_kickoff = context.game.kicking_team == context.team_id
    positions = READY_OUR_POSITIONS if is_our_kickoff else READY_OPP_POSITIONS
    Player.walk_to_slots(players, positions, Action.READY, face=0.0)


def _draw_teammate_markers(players: list[Player]) -> None:
    from .framework import debugdraw

    red = (1.0, 0.2, 0.2)
    for p in players:
        if p.pose is None:
            continue
        if p.is_kicking:
            debugdraw.cube(p.pose.x, p.pose.y, rgb=red, scale=0.38, ns="teammate")
        else:
            debugdraw.point(p.pose.x, p.pose.y, rgb=red, scale=0.3, ns="teammate")
        kick_tag = " [KICK]" if p.is_kicking else ""
        label = f"{p.id}:{p.action}{kick_tag}"
        debugdraw.text(p.pose.x, p.pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player]) -> None:
    from .framework import debugdraw

    ball = context.ball

    if ball is None:
        return

    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    for p in players:
        if p.pose is None:
            continue
        d = dist(p.pose.x, p.pose.y, ball.x, ball.y)
        debugdraw.text(
            p.pose.x + 0.3, p.pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        if r.pose is None:
            continue
        d = dist(r.pose.x, r.pose.y, ball.x, ball.y)
        debugdraw.text(
            r.pose.x + 0.3, r.pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )