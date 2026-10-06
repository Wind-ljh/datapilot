"""多轮对话数据分析端到端验证（离线 mock，走真实 Streamlit 渲染路径）。

用 AppTest 驱动 app/streamlit_app.py，验证 TASK E 三条验收：
1. 追问：问答 → 追问 → 结构化上下文随下一次提问传入（context 已保存）；
2. 新对话：彻底清空消息与上下文、换取新 id（会话隔离）；
3. 历史：打开历史会话 → 恢复其消息 + 上下文 → 可继续分析。

会话持久化写入临时目录，不污染仓库 data/conversations.json。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ["DATAPILOT_MOCK"] = "1"
os.environ["DATAPILOT_CONVERSATIONS_PATH"] = str(
    Path(tempfile.mkdtemp()) / "conversations.json"
)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _find_button(at, substr: str):
    for b in at.button:
        if substr in str(b.label):
            return b
    return None


def _exceptions(at) -> list[str]:
    return [str(e.value) for e in at.exception]


def main() -> int:
    fails = 0

    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()

    # 初始状态：空消息、无上下文
    assert at.session_state["messages"] == [], "初始消息应为空"
    assert at.session_state["context"] is None, "初始上下文应为空"
    id1 = at.session_state["current_id"]

    # ---- 第 1 问：建立分析上下文
    q1 = "各品类的销售额是多少？"
    at.chat_input[0].set_value(q1)
    at.run()
    msgs_after_q1 = len(at.session_state["messages"])
    ctx_after_q1 = at.session_state["context"]
    print(
        f"[Q1] {q1} -> messages={msgs_after_q1} "
        f"context={'set' if ctx_after_q1 else 'None'} exceptions={_exceptions(at)}"
    )
    if msgs_after_q1 != 2:
        fails += 1
    if ctx_after_q1 is None:
        fails += 1

    # ---- 第 2 问：指代追问（应携带上一轮上下文，无异常）
    q2 = "只看前3个"
    at.chat_input[0].set_value(q2)
    at.run()
    msgs_after_q2 = len(at.session_state["messages"])
    ctx_after_q2 = at.session_state["context"]
    print(
        f"[Q2] {q2} -> messages={msgs_after_q2} "
        f"context={'set' if ctx_after_q2 else 'None'} exceptions={_exceptions(at)}"
    )
    if msgs_after_q2 != msgs_after_q1 + 2:
        fails += 1
    if ctx_after_q2 is None:
        fails += 1
    if _exceptions(at):
        fails += 1

    # ---- 新对话：清空上下文 + 新 id
    new_btn = _find_button(at, "新对话")
    assert new_btn is not None, "应存在『新对话』按钮"
    new_btn.click()
    at.run()
    id2 = at.session_state["current_id"]
    print(
        f"[NEW] messages={len(at.session_state['messages'])} "
        f"context={at.session_state['context']} new_id={'yes' if id2 != id1 else 'no'}"
    )
    if at.session_state["messages"] != []:
        fails += 1
    if at.session_state["context"] is not None:
        fails += 1
    if id2 == id1:
        fails += 1

    # ---- 历史：打开上一会话，恢复消息 + 上下文
    open_btn = _find_button(at, q1)  # 会话标题 = 第一条用户问题
    assert open_btn is not None, "侧边栏应存在上一会话的入口"
    open_btn.click()
    at.run()
    restored_msgs = len(at.session_state["messages"])
    restored_ctx = at.session_state["context"]
    restored_id = at.session_state["current_id"]
    print(
        f"[RESTORE] messages={restored_msgs} "
        f"context={'set' if restored_ctx else 'None'} "
        f"id_match={'yes' if restored_id == id1 else 'no'}"
    )
    if restored_msgs != msgs_after_q2:
        fails += 1
    if restored_ctx is None:
        fails += 1
    if restored_id != id1:
        fails += 1

    print(f"\nTOTAL fails={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
