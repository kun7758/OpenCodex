"""
调试脚本：在 PyCharm 中右键 Run/Debug 即可运行，支持断点调试。
等价于 `uv run opennova plan "..."` 或 `uv run opennova run "..."` 的核心逻辑。

使用方式：
    1. 在 PyCharm 中打开此文件
    2. 修改下方 TASK 和 MODE
    3. 右键 → Debug 'debug_plan_flow'
    4. 在 agent.py / loop.py / execution.py 等处打断点即可
"""

import asyncio
import sys
from pathlib import Path

# 确保项目 src 在 sys.path 中，PyCharm 直接运行时也能找到 opennova 包
# 脚本位于 tests/ 下，需要向上一层找到项目根目录的 src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opennova.config import load_config, validate_config
from opennova.logging_config import get_logger, setup_logging
from opennova.providers.base import StreamChunk
from opennova.runtime.agent import AgentRuntime
from opennova.runtime.bootstrap import RuntimeBootstrapProfile
from opennova.runtime.state import Plan
from opennova.tools.base import ToolResult

_LOGGER = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 在这里修改你的任务和模式
# ═══════════════════════════════════════════════════════════════════
TASK = "帮我使用python实现一个文件上传下载的功能，实现形式为对外暴露两个http接口，一个文件上传，一个文件下载，上传文件后返回文件的访问路径，下载时使用该路径下载文件，保存为一个.py文件"
MODE = "plan"       # "plan" = 先生成计划再执行，"act" = 直接执行
STREAM = True       # 是否流式输出模型回复
EXECUTE_PLAN = True  # plan 模式下，生成计划后是否自动执行（模拟输入 y）
MAX_ITERATIONS = 30  # 最大迭代次数，调试时可适当调大
# ═══════════════════════════════════════════════════════════════════


async def main() -> None:
    # 1. 加载配置（自动读取 .env 和 ~/.opennova/config.yaml）
    config = load_config()

    # 2. 初始化日志系统
    logging_config = config.get_logging_config()
    setup_logging(logging_config)

    errors = validate_config(config)
    if errors:
        _LOGGER.error("配置错误：")
        for e in errors:
            _LOGGER.error("  - %s", e)
        return

    # 覆盖最大迭代次数，方便调试
    config.set("agent.max_iterations", MAX_ITERATIONS)

    # 2. 创建 AgentRuntime
    agent = AgentRuntime(config, bootstrap_profile=RuntimeBootstrapProfile.HEADLESS)

    # 3. 注册回调（控制台输出）
    def on_thought(thought: str) -> None:
        print(f"\n💭 {thought}")

    def on_action(tool_name: str, args: dict) -> None:
        args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in args.items())
        print(f"⚙️  {tool_name}({args_str})")

    def on_result(result: ToolResult) -> None:
        if result.success:
            print(f"✅ {result.output[:200] if result.output else 'Done'}")
        else:
            print(f"❌ Error: {result.error}")

    def on_stream(chunk: StreamChunk) -> None:
        if chunk.content:
            print(chunk.content, end="", flush=True)

    def on_plan(plan: Plan, plan_file_path: str | None = None) -> None:
        print(f"\n📋 生成计划：{len(plan.steps)} 个步骤")
        for step in plan.steps:
            print(f"   {step.id}: {step.description}")
        if plan_file_path:
            print(f"   保存路径: {plan_file_path}")

    agent.register_callback("thought", on_thought)
    agent.register_callback("action", on_action)
    agent.register_callback("result", on_result)
    agent.register_callback("stream", on_stream)
    agent.register_callback("plan", on_plan)

    # 4. 运行任务
    try:
        _LOGGER.info("模式: %s", MODE)
        _LOGGER.info("任务: %s", TASK)
        _LOGGER.info("-" * 60)

        result = await agent.run(TASK, mode=MODE, stream=STREAM)

        _LOGGER.info("=" * 60)
        _LOGGER.info("结果: %s", result)

        # 5. plan 模式下自动执行（模拟用户输入 y）
        if MODE == "plan" and EXECUTE_PLAN and agent.state.current_plan:
            _LOGGER.info("自动执行计划...")
            agent.state.mark_plan_approved()
            exec_result = await agent.execute_approved_plan(stream=STREAM)
            _LOGGER.info("执行结果: %s", exec_result)

    except KeyboardInterrupt:
        _LOGGER.warning("任务被中断。")
    except Exception as e:
        _LOGGER.error("错误: %s: %s", type(e).__name__, str(e), exc_info=True)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
