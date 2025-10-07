import uuid
from dataclasses import dataclass
from typing import Protocol
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent_core import agent


@dataclass
class SessionState:
    """セッション状態を管理する値オブジェクト

    目的: UI状態データをカプセル化し、安全な状態管理を実現
    """
    messages: list[dict]
    waiting_for_approval: bool
    final_result: str | None
    thread_id: str | None
    tool_info: dict | None = None

    @staticmethod
    def create_initial() -> 'SessionState':
        """初期状態を生成

        Returns:
            SessionState: 初期化されたセッション状態
        """
        return SessionState(
            messages=[],
            waiting_for_approval=False,
            final_result=None,
            thread_id=None,
            tool_info=None
        )

    def reset(self) -> 'SessionState':
        """状態をリセット

        Returns:
            SessionState: リセットされた新しいセッション状態
        """
        return SessionState.create_initial()


class SessionStateManager:
    """Streamlitセッション状態を管理する責務を持つクラス

    目的: st.session_stateへのアクセスをカプセル化し、
         状態の初期化とリセットロジックを統一
    """

    def __init__(self):
        """セッション状態を初期化"""
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """セッション状態が初期化されていることを保証"""
        if 'messages' not in st.session_state:
            initial_state = SessionState.create_initial()
            st.session_state.messages = initial_state.messages
            st.session_state.waiting_for_approval = initial_state.waiting_for_approval
            st.session_state.final_result = initial_state.final_result
            st.session_state.thread_id = initial_state.thread_id
            st.session_state.tool_info = initial_state.tool_info

    def reset(self) -> None:
        """セッション状態をリセット"""
        reset_state = SessionState.create_initial()
        st.session_state.messages = reset_state.messages
        st.session_state.waiting_for_approval = reset_state.waiting_for_approval
        st.session_state.final_result = reset_state.final_result
        st.session_state.thread_id = reset_state.thread_id
        st.session_state.tool_info = reset_state.tool_info

    def add_message(self, role: str, content: str) -> None:
        """メッセージを追加

        Args:
            role: メッセージの役割(user/assistant/system)
            content: メッセージ内容
        """
        st.session_state.messages.append({"role": role, "content": content})

    def set_waiting_approval(self, waiting: bool) -> None:
        """承認待ち状態を設定

        Args:
            waiting: 承認待ちかどうか
        """
        st.session_state.waiting_for_approval = waiting

    def set_tool_info(self, tool_info: dict) -> None:
        """ツール情報を設定

        Args:
            tool_info: ツール情報辞書
        """
        st.session_state.tool_info = tool_info

    def set_final_result(self, result: str) -> None:
        """最終結果を設定

        Args:
            result: 最終結果文字列
        """
        st.session_state.final_result = result

    def set_thread_id(self, thread_id: str) -> None:
        """スレッドIDを設定

        Args:
            thread_id: スレッドID
        """
        st.session_state.thread_id = thread_id

    @property
    def messages(self) -> list[dict]:
        """メッセージリストを取得"""
        return st.session_state.messages

    @property
    def waiting_for_approval(self) -> bool:
        """承認待ち状態を取得"""
        return st.session_state.waiting_for_approval

    @property
    def tool_info(self) -> dict | None:
        """ツール情報を取得"""
        return st.session_state.tool_info

    @property
    def final_result(self) -> str | None:
        """最終結果を取得"""
        return st.session_state.final_result

    @property
    def thread_id(self) -> str | None:
        """スレッドIDを取得"""
        return st.session_state.thread_id


class AgentStreamProcessor:
    """エージェントのストリーミング処理を管理する責務を持つクラス

    目的: エージェント実行結果のストリーム処理をカプセル化
    """

    def __init__(self, session_manager: SessionStateManager):
        """
        Args:
            session_manager: セッション状態管理
        """
        self._session_manager = session_manager

    def run(self, input_data: list[HumanMessage] | Command) -> None:
        """エージェントを実行し、結果をストリーミング処理

        Args:
            input_data: エージェントへの入力データ
        """
        config = {"configurable": {"thread_id": self._session_manager.thread_id}}

        with st.spinner("処理中...", show_time=True):
            for chunk in agent.stream(input_data, stream_mode="updates", config=config):
                self._process_chunk(chunk)
            st.rerun()

    def _process_chunk(self, chunk: dict) -> None:
        """チャンクを処理

        Args:
            chunk: ストリーミングされたチャンクデータ
        """
        for task_name, result in chunk.items():
            print("task_name:", task_name, "result:", result)

            if task_name == "__interrupt__":
                self._handle_interrupt(result)
            elif task_name == "agent":
                self._handle_agent_result(result)
            elif task_name == "invoke_llm":
                self._handle_llm_result(result)
            elif task_name == "use_tool":
                self._handle_tool_execution()

    def _handle_interrupt(self, result: list) -> None:
        """割り込み(承認待ち)を処理"""
        self._session_manager.set_tool_info(result[0].value)
        self._session_manager.set_waiting_approval(True)

    def _handle_agent_result(self, result) -> None:
        """エージェント最終結果を処理"""
        self._session_manager.set_final_result(result.content)

    def _handle_llm_result(self, result) -> None:
        """LLM推論結果を処理"""
        if isinstance(result.content, list):
            for content in result.content:
                if content["type"] == "text":
                    self._session_manager.add_message("assistant", content["text"])

    def _handle_tool_execution(self) -> None:
        """ツール実行を処理"""
        self._session_manager.add_message("system", "ツール実行")


class UserFeedbackCollector:
    """ユーザーフィードバック収集を管理する責務を持つクラス

    目的: ツール承認/拒否のUI表示とフィードバック収集をカプセル化
    """

    APPROVE_FEEDBACK = "APPROVE"
    DENY_FEEDBACK = "DENY"

    def collect(self) -> str | None:
        """承認/拒否フィードバックを収集

        Returns:
            str | None: "APPROVE"/"DENY" または None(未選択)
        """
        approve_column, deny_column = st.columns(2)

        with approve_column:
            if st.button("承認"):
                return self.APPROVE_FEEDBACK

        with deny_column:
            if st.button("拒否"):
                return self.DENY_FEEDBACK

        return None


class MessageDisplayRenderer:
    """メッセージ表示を管理する責務を持つクラス

    目的: チャットメッセージのレンダリングロジックをカプセル化
    """

    def render(self, messages: list[dict]) -> None:
        """メッセージリストを表示

        Args:
            messages: 表示するメッセージリスト
        """
        for message in messages:
            self._render_single_message(message)

    def _render_single_message(self, message: dict) -> None:
        """単一メッセージを表示

        Args:
            message: 表示するメッセージ
        """
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)


class ToolApprovalRenderer:
    """ツール承認UIを表示する責務を持つクラス

    目的: ツール承認待ちUIのレンダリングをカプセル化
    """

    def __init__(self, feedback_collector: UserFeedbackCollector):
        """
        Args:
            feedback_collector: フィードバック収集
        """
        self._feedback_collector = feedback_collector

    def render_and_collect_feedback(self, tool_info: dict) -> str | None:
        """ツール承認UIを表示しフィードバックを収集

        Args:
            tool_info: ツール情報

        Returns:
            str | None: ユーザーのフィードバック結果
        """
        st.info(tool_info["args"])

        if tool_info["name"] == "write_file" and "html" in tool_info:
            with st.container():
                st.html(tool_info["html"])

        return self._feedback_collector.collect()


class ResearchAgentUI:
    """調査エージェントUIのオーケストレーションを管理するクラス

    目的: UI全体の表示フローと各コンポーネントの協調動作を管理
    """

    def __init__(
        self,
        session_manager: SessionStateManager,
        stream_processor: AgentStreamProcessor,
        message_renderer: MessageDisplayRenderer,
        approval_renderer: ToolApprovalRenderer
    ):
        """
        Args:
            session_manager: セッション状態管理
            stream_processor: エージェントストリーミング処理
            message_renderer: メッセージ表示
            approval_renderer: ツール承認UI表示
        """
        self._session_manager = session_manager
        self._stream_processor = stream_processor
        self._message_renderer = message_renderer
        self._approval_renderer = approval_renderer

    def run(self) -> None:
        """UIアプリケーションを実行"""
        st.title("AIエージェントデモアプリ")

        self._message_renderer.render(self._session_manager.messages)

        if self._session_manager.waiting_for_approval and self._session_manager.tool_info:
            self._handle_tool_approval()
        elif self._session_manager.final_result and not self._session_manager.waiting_for_approval:
            self._display_final_result()

        if not self._session_manager.waiting_for_approval:
            self._handle_user_input()
        else:
            st.info("ツールの使用承認を待っています。上記のボタンを押してください。")

    def _handle_tool_approval(self) -> None:
        """ツール承認処理"""
        feedback_result = self._approval_renderer.render_and_collect_feedback(
            self._session_manager.tool_info
        )

        if feedback_result:
            st.chat_message("user").write(feedback_result)
            self._session_manager.add_message("user", feedback_result)
            self._session_manager.set_waiting_approval(False)
            self._stream_processor.run(Command(resume=feedback_result))

    def _display_final_result(self) -> None:
        """最終結果を表示"""
        st.subheader("最終結果")
        st.success(self._session_manager.final_result)

    def _handle_user_input(self) -> None:
        """ユーザー入力を処理"""
        user_input = st.chat_input("入力してください。")

        if user_input:
            self._session_manager.reset()
            self._session_manager.set_thread_id(str(uuid.uuid4()))

            st.chat_message("user").write(user_input)
            self._session_manager.add_message("user", user_input)

            messages = [HumanMessage(content=user_input)]
            self._stream_processor.run(messages)


def main():
    """メイン関数: アプリケーションの初期化と実行"""
    session_manager = SessionStateManager()
    stream_processor = AgentStreamProcessor(session_manager)
    message_renderer = MessageDisplayRenderer()
    feedback_collector = UserFeedbackCollector()
    approval_renderer = ToolApprovalRenderer(feedback_collector)

    ui = ResearchAgentUI(
        session_manager=session_manager,
        stream_processor=stream_processor,
        message_renderer=message_renderer,
        approval_renderer=approval_renderer
    )

    ui.run()


if __name__ == "__main__":
    main()
