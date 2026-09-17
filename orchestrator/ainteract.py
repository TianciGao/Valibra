"""BIRD-Interact ADK 编排器——a-interact 智能体流程。

此版本将完整的工具调用循环交由运行在 6000 端口、
基于 ADK 实现的系统智能体服务负责。

编排器只负责：
1. 初始化数据库环境服务和用户模拟器服务
2. 在系统智能体服务中初始化一个智能体会话
3. 将初始用户请求发送一次
4. 读取最终的会话状态，以获取评测指标
"""

import argparse
import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import httpx
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import settings
from valibra_agent.evaluation import export_valibra_result
from valibra_agent.runtime_profile import valibra_execution_profile
from valibra_agent.sql_grounding.main_entry_carrier import (
    MAIN_EXECUTION_ENVELOPES_KEY,
    build_main_execution_envelopes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s") # 设置日志格式和级别
logger = logging.getLogger(__name__)

SYSTEM_AGENT_URL = f"http://localhost:{settings.system_agent_port}"
USER_SIM_URL = f"http://localhost:{settings.user_sim_port}"
DB_ENV_URL = f"http://localhost:{settings.db_env_port}"


async def _post(url: str, payload: dict, timeout: float = 120.0) -> dict: # 定义一个异步函数，用于向指定 URL 发送 POST 请求，并返回响应的 JSON 数据
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client: # 创建一个异步 HTTP 客户端，设置请求超时时间和信任环境变量
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def _get(url: str, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def calculate_initial_budget(task_data: Dict[str, Any]) -> float: # 定义一个函数，用于计算每个任务的初始预算（以 bird-coins 为单位），根据任务数据中的用户查询歧义和知识歧义来计算
    """a-interact budget in bird-coins (per task, paper Section 3.2). 

    Formula: 6 + 2 * m_amb + 2 * patience
      - 6 = ENV_INTERACT(3) + SUBMIT(3) base budget
      - 2 * m_amb = one ask_user (cost=2) per ambiguity point
      - 2 * patience = extra exploration tolerance
      - patience=3 in config = patience_budget=6 in reference (we multiply by 2)
    """
    critical = len(task_data.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    knowledge = len(task_data.get("knowledge_ambiguity", []))
    m_amb = critical + knowledge
    return 6.0 + 2.0 * m_amb + 2.0 * settings.patience


async def init_task_on_services(task_id: str, task_data: dict): # 定义一个异步函数，用于在数据库环境服务和用户模拟器服务中初始化任务
    payload = {
        "task_id": task_id, 
        "task_data": {**task_data, "_interact_mode": "a-interact"}, # 将任务数据和交互模式封装为一个字典，传递给服务
    }
    await _post(f"{DB_ENV_URL}/init_task", payload)  # 定义一个异步函数，用于在数据库环境服务中初始化任务
    await _post(f"{USER_SIM_URL}/init_task", payload)  # 定义一个异步函数，用于在用户模拟器服务中初始化任务
    logger.info("  [%s] Services initialized", task_id)  # 记录服务初始化完成的日志


async def init_agent_session(task_id: str, task_data: dict, budget: float): # 定义一个异步函数，用于在系统智能体服务中初始化一个智能体会话，并设置初始状态
    state = {
        "task_id": task_id,
        "db_name": task_data["selected_database"],
        "user_query": task_data.get("amb_user_query", ""),
        "current_phase": 1,
        "budget_remaining": budget,
        "initial_budget": budget,
        "total_reward": 0.0,
        "dialogue_history": [],
        "tool_trajectory": [],
        "adk_events": [],
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        MAIN_EXECUTION_ENVELOPES_KEY: build_main_execution_envelopes(task_data),
    } # 定义一个字典，包含任务 ID、数据库名称、用户查询、当前阶段、剩余预算、初始预算、总奖励、对话历史、工具轨迹、ADK 事件、阶段完成状态和任务完成状态等信息
    return await _post(
        f"{SYSTEM_AGENT_URL}/init_session",
        {"task_id": task_id, "mode": "a-interact", "state": state, "reset": True},
        timeout=30.0,
    ) # 定义一个异步函数，用于向系统智能体服务发送 POST 请求，初始化一个智能体会话，并传递任务 ID、交互模式、初始状态和重置标志等信息


async def run_agent_session(task_id: str, message: str): # 定义一个异步函数，用于在系统智能体服务中运行一个智能体会话，并发送初始用户请求
    return await _post(
        f"{SYSTEM_AGENT_URL}/run_session",
        {"task_id": task_id, "mode": "a-interact", "message": message},
        timeout=1800.0,
    )


async def cleanup_task_service(task_id: str): # 定义一个异步函数，用于在数据库环境服务中清理任务
    try:
        await _post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
    except Exception as e:
        logger.warning("Cleanup failed for %s: %s", task_id, e)


async def run_single_task(task_data: dict) -> Dict[str, Any]: # 定义一个异步函数，用于运行单个任务，并返回任务结果
    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    logger.info("Starting task: %s (db: %s)", instance_id, db_name)
    start_time = time.time()

    await init_task_on_services(instance_id, task_data) # 初始化数据库环境服务和用户模拟器服务中的任务

    try:
        initial_budget = calculate_initial_budget(task_data)  # 计算每个任务的初始预算（以 bird-coins 为单位），根据任务数据中的用户查询歧义和知识歧义来计算
        await init_agent_session(instance_id, task_data, initial_budget) # 在系统智能体服务中初始化一个智能体会话，并设置初始状态

        initial_message = (
            f"Database: {db_name}\n"
            f"Task ID: {instance_id}\n\n"
            f"User Query:\n{task_data.get('amb_user_query', '')}\n\n"
            f"You have a budget of {initial_budget:.1f} bird-coins. "
            f"Use your tools to explore the database, clarify ambiguities with the user, "
            f"and submit your final SQL efficiently."
        ) # 定义一个字符串，包含数据库名称、任务 ID、用户查询和初始预算等信息，作为初始用户请求发送给系统智能体服务

        run_result = await run_agent_session(instance_id, initial_message) # 在系统智能体服务中运行一个智能体会话，并发送初始用户请求，返回运行结果
        state = run_result.get("state", {}) # 获取运行结果中的状态信息，包含任务 ID、数据库名称、用户查询、当前阶段、剩余预算、初始预算、总奖励、对话历史、工具轨迹、ADK 事件、阶段完成状态和任务完成状态等信息
        infrastructure_error = state.get("_infrastructure_error")
        if infrastructure_error:
            raise RuntimeError(str(infrastructure_error))
        try:
            user_simulator_audit = await _get(f"{USER_SIM_URL}/audit/{instance_id}")
        except Exception as exc:
            logger.warning("User-simulator audit unavailable for %s: %s", instance_id, exc)
            user_simulator_audit = {"error": str(exc), "llm_calls": [], "token_usage": {}}
        elapsed = time.time() - start_time # 计算任务运行的总耗时，单位为秒

        trajectory = state.get("tool_trajectory", [])
        phase1_sql = [
            event.get("args", {}).get("sql", "")
            for event in trajectory
            if event.get("tool") == "submit_sql" and event.get("phase", 1) == 1
        ]
        phase2_sql = [
            event.get("args", {}).get("sql", "")
            for event in trajectory
            if event.get("tool") == "submit_sql" and event.get("phase", 1) == 2
        ]
        system_usage = state.get("system_agent_token_usage", {})
        user_usage = user_simulator_audit.get("token_usage", {})
        token_usage = {
            "system_agent": system_usage,
            "user_simulator": user_usage,
            "combined": {
                field: int(system_usage.get(field, 0) or 0)
                + int(user_usage.get(field, 0) or 0)
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cached_tokens",
                    "reasoning_tokens",
                    "tool_prompt_tokens",
                )
            },
        }

        result = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": db_name,
            "phase1_passed": state.get("phase1_completed", False),
            "phase2_passed": state.get("phase2_completed", False),
            "has_follow_up": bool(task_data.get("follow_up") and task_data["follow_up"].get("sol_sql")),
            "total_reward": state.get("total_reward", 0.0),
            "elapsed_seconds": elapsed,
            "initial_budget": initial_budget,
            "budget_used": initial_budget - max(0, state.get("budget_remaining", initial_budget)),
            "budget_remaining": max(0, state.get("budget_remaining", initial_budget)),
            "system_agent_model": settings.system_agent_model,
            "user_simulator_model": settings.user_sim_model,
            "user_simulator_prompt_version": settings.prompt_version,
            "initial_message": initial_message,
            "subtask_1_predicted_sql": [phase1_sql[-1]] if phase1_sql else [],
            "subtask_2_predicted_sql": [phase2_sql[-1]] if phase2_sql else [],
            "all_submitted_sql": {
                "phase1": phase1_sql,
                "phase2": phase2_sql,
            },
            "prompt_flow": state.get("system_agent_llm_calls", []),
            "token_usage": token_usage,
            "dialogue_history": state.get("dialogue_history", []),
            "tool_trajectory": trajectory,
            "adk_events": state.get("adk_events", []),
            "user_simulator_audit": user_simulator_audit,
            "final_response": run_result.get("response", ""),
        } # 定义一个字典，包含任务 ID、数据库名称、阶段完成状态、是否有后续任务、总奖励、耗时、预算使用情况、剩余预算、对话历史、工具轨迹、ADK 事件和最终响应等信息
        # Research keeps the additive private export.  Leaderboard preserves the
        # frozen Official result schema and never appends result["valibra"].
        if valibra_execution_profile() == "research":
            try:
                result["valibra"] = export_valibra_result(
                    state,
                    user_simulator_audit=user_simulator_audit,
                )
            except Exception as exc:
                result["valibra"] = {
                    "export_status": "failed",
                    "error_type": type(exc).__name__[:128],
                }
        logger.info( # 记录任务完成的日志，包含任务 ID、总奖励、预算使用情况和耗时等信息
            "Task %s done. Reward: %.2f, Budget used: %.1f, Time: %.1fs",
            instance_id,
            result["total_reward"],
            result["budget_used"],
            elapsed,
        )
        return result
    finally:
        await cleanup_task_service(instance_id)


async def run_evaluation(data_path: str, output_path: str, limit: int = None): # 定义一个异步函数，用于运行评测任务，并将结果保存到指定的输出路径
    tasks = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    logger.info("A-Interact: Evaluating %d tasks", len(tasks))

    results = []
    total_reward = 0.0
    p1_count = 0
    p2_count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for i, td in enumerate(tasks): # 遍历所有任务数据，依次运行每个任务，并记录结果
        logger.info("=== Task %d/%d: %s ===", i + 1, len(tasks), td["instance_id"])
        try:
            r = await run_single_task(td)
            results.append(r)
            total_reward += r["total_reward"]
            if r["phase1_passed"]:
                p1_count += 1
            if r["phase2_passed"]:
                p2_count += 1
        except Exception as e:
            logger.error("Error: %s: %s", td["instance_id"], e)
            traceback.print_exc()
            results.append({"task_id": td["instance_id"], "error": str(e), "total_reward": 0})

        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            n = len(results)
            output = {
                "mode": "a-interact",
                "metrics": {
                    "total_tasks": n,
                    "total_reward": total_reward,
                    "average_reward": total_reward / n if n else 0,
                    "phase1_rate": p1_count / n if n else 0,
                    "phase2_rate": p2_count / n if n else 0,
                    "phase1_count": p1_count,
                    "phase2_count": p2_count,
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

    n = len(tasks)
    if n:
        logger.info(
            "\nDone! Tasks: %d, Avg Reward: %.4f, P1: %d/%d (%.1f%%), P2: %d/%d (%.1f%%)",
            n,
            total_reward / n,
            p1_count,
            n,
            p1_count / n * 100,
            p2_count,
            n,
            p2_count / n * 100,
        ) # 记录评测完成的日志，包含任务总数、平均奖励、阶段 1 和阶段 2 的通过率等信息


def main(): # 定义主函数，用于解析命令行参数，并运行评测任务
    parser = argparse.ArgumentParser(description="BIRD-Interact a-interact evaluation")
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--output", default="results/eval_ainteract.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.data, args.output, args.limit))


if __name__ == "__main__": 
    main()
