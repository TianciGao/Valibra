"""Unified parallel evaluation runner for BIRD-Interact benchmark."""
# 读取 bird_interact_data.jsonl 的每一条任务
# 根据命令行 --mode, --data, --output, --limit, --concurrency 参数，选择不同的评估模式和任务数据
# shared/config.py 中的 Settings 类提供了配置参数，包括数据库连接、服务端口、模型信息等

import asyncio
import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # 将 BIRD-Interact-ADK 的根目录添加到 sys.path，以便导入 shared/config.py 中的 Settings 类

from shared.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s") # 设置日志格式和级别
logger = logging.getLogger(__name__) # 创建一个日志记录器对象


async def run_parallel_evaluation(  # 定义一个异步函数，用于并行运行评估任务
    tasks: List[dict],
    run_single_task: Callable[[dict], Awaitable[Dict[str, Any]]],
    output_path: str,
    concurrency: int = 5, # 并发数，默认值为 5
    mode: str = "a-interact", # python -m orchestrator.runner 默认 mode="a-interact"  (可选: "c-interact", "oracle"
):
    semaphore = asyncio.Semaphore(concurrency) # 创建一个信号量对象，用于限制并发数 
    results: List[Dict[str, Any]] = [] # 用于存储每个任务的结果
    results_lock = asyncio.Lock() # 用于保护对 results 列表的并发访问
    total_reward = 0.0 # 初始化总奖励为 0.0
    p1_count = 0 # 初始化阶段 1 通过的任务数为 0
    p2_count = 0 # 初始化阶段 2 通过的任务数为 0
    completed = 0 # 初始化已完成的任务数为 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) # 创建输出文件的父目录（如果不存在）

    async def _save(): # 定义一个异步函数，用于保存当前的评估结果到输出文件
        n = len(results)
        if n == 0:
            return
        output = {
            "mode": mode,
            "metrics": {
                "total_tasks": n,
                "total_reward": total_reward,
                "average_reward": total_reward / n,
                "phase1_rate": p1_count / n,
                "phase2_rate": p2_count / n,
                "phase1_count": p1_count,
                "phase2_count": p2_count,
            },
            "results": results,
        }
        with open(output_path, "w") as f: # 将评估结果写入输出文件
            json.dump(output, f, indent=2, default=str)  

    async def _run_one(i: int, td: dict): # 定义一个异步函数，用于运行单个任务
        nonlocal total_reward, p1_count, p2_count, completed # 声明 total_reward、p1_count、p2_count 和 completed 为非局部变量，以便在函数内部修改它们的值
        instance_id = td["instance_id"] 
        async with semaphore: 
            logger.info("=== Task %d/%d: %s ===", i + 1, len(tasks), instance_id)
            try:
                r = await run_single_task(td)
            except Exception as e:
                logger.error("Error: %s: %s", instance_id, e)
                traceback.print_exc()
                r = {"task_id": instance_id, "error": str(e), "total_reward": 0}

        async with results_lock: # 用于保护对 results 列表的并发访问，确保在多任务并发执行时不会出现数据竞争
            results.append(r) 
            total_reward += r.get("total_reward", 0)
            if r.get("phase1_passed"):
                p1_count += 1
            if r.get("phase2_passed"):
                p2_count += 1
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                await _save()

    await asyncio.gather(*[_run_one(i, td) for i, td in enumerate(tasks)]) # 并行运行所有任务，并等待它们全部完成
    await _save() # 保存最终的评估结果到输出文件

    n = len(tasks) # 获取任务总数
    if n:
        logger.info(
            "\nDone! Tasks: %d, Avg Reward: %.4f, P1: %d/%d (%.1f%%), P2: %d/%d (%.1f%%)",
            n, total_reward / n, p1_count, n, p1_count / n * 100,
            p2_count, n, p2_count / n * 100,
        ) # 记录评估结果的日志信息，包括任务总数、平均奖励、阶段 1 和阶段 2 的通过率等


def load_tasks(data_path: str, limit: int = None) -> List[dict]: # 定义一个函数，用于加载任务数据
    tasks = [] # 初始化一个空列表，用于存储任务数据
    with open(data_path) as f: # 读取任务数据文件
        for line in f: 
             if line.strip():
                tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    return tasks


async def run_oracle_task(task_data: dict) -> Dict[str, Any]: # 定义一个异步函数，用于运行 Oracle 管线测试任务
    """直接把标准答案 SQL 提交给评测器，不经过任何大模型，用来检查整条评测流水线是否正常，见README.md的“Oracle管线测试”部分""" 
    import httpx 
    db_env = f"http://localhost:{settings.db_env_port}"
    user_sim = f"http://localhost:{settings.user_sim_port}"

    async def _post(url, payload, timeout=60.0): # 定义一个异步函数，用于发送 HTTP POST 请求
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c: 
            r = await c.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    instance_id = task_data["instance_id"] 
    sol_sql = task_data.get("sol_sql", [])
    if isinstance(sol_sql, str): # 如果 sol_sql 是字符串类型，则将其转换为列表类型
        sol_sql = [sol_sql]
    fu = task_data.get("follow_up", {})
    fu_sql = fu.get("sol_sql", [])
    if isinstance(fu_sql, str): # 如果 fu_sql 是字符串类型，则将其转换为列表类型
        fu_sql = [fu_sql]
    has_follow_up = bool(fu and fu_sql)

    await _post(f"{db_env}/init_task", {  # 初始化任务，向数据库环境服务发送 HTTP POST 请求，传递任务 ID 和任务数据
        "task_id": instance_id,
        "task_data": {**task_data, "_interact_mode": "a-interact"},
    })

    try:
        p1_passed = False # 初始化阶段 1 是否通过的标志为 False
        p2_passed = False # 初始化阶段 2 是否通过的标志为 False
        total_reward = 0.0 # 初始化总奖励为 0.0

        if sol_sql:
            r1 = await _post(f"{db_env}/submit", {"task_id": instance_id, "sql": sol_sql[0]})
            p1_passed = r1.get("passed", False)
            if p1_passed:
                total_reward += r1.get("reward", 0.0)

            if p1_passed and has_follow_up:
                try:
                    await _post(f"{user_sim}/init_task", {"task_id": instance_id, "task_data": task_data})
                    await _post(f"{user_sim}/phase_transition", {"task_id": instance_id})
                except Exception:
                    pass
                r2 = await _post(f"{db_env}/submit", {"task_id": instance_id, "sql": fu_sql[0]})
                p2_passed = r2.get("passed", False)
                if p2_passed:
                    total_reward += r2.get("reward", 0.0)

        return {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": task_data["selected_database"],
            "phase1_passed": p1_passed,
            "phase2_passed": p2_passed,
            "has_follow_up": has_follow_up,
            "total_reward": total_reward,
        }
    finally:
        try:
            await _post(f"{db_env}/cleanup_task", {"task_id": instance_id})
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="BIRD-Interact parallel evaluation") # 创建一个命令行参数解析器对象
    parser.add_argument("--mode", choices=["a-interact", "c-interact", "oracle"], default="a-interact") # 添加一个命令行参数 --mode，用于指定评估模式，默认值为 "a-interact"
    parser.add_argument("--data", default=settings.data_path) # 添加一个命令行参数 --data，用于指定任务数据文件路径，默认值为 settings.data_path
    parser.add_argument("--output", default=None) # 添加一个命令行参数 --output，用于指定输出文件路径，默认值为 None
    parser.add_argument("--limit", type=int, default=None) # 添加一个命令行参数 --limit，用于限制评估任务的数量，默认值为 None
    parser.add_argument("--concurrency", type=int, default=5) # 添加一个命令行参数 --concurrency，用于指定并发数，默认值为 5
    args = parser.parse_args() # 解析命令行参数，并将结果存储在 args 对象中

    output = args.output or f"results/eval_{args.mode.replace('-', '_')}.json" # 如果未指定输出文件路径，则根据评估模式生成默认的输出文件路径

    if args.mode == "oracle":
        run_single_task = run_oracle_task
    elif args.mode == "a-interact":
        from orchestrator.ainteract import run_single_task
    else:
        from orchestrator.cinteract import run_single_task

    tasks = load_tasks(args.data, args.limit)
    logger.info("%s: Evaluating %d tasks with concurrency=%d", args.mode, len(tasks), args.concurrency)

    asyncio.run(run_parallel_evaluation(
        tasks=tasks,
        run_single_task=run_single_task,
        output_path=output,
        concurrency=args.concurrency,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
