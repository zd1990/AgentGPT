from revChatGPT.V1 import Chatbot
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.base import BaseCallbackHandler


class CustomLLM(LLM):
    model: str = "gpt-4"
    chatbot: Chatbot = None
    access_token: str = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[BaseCallbackHandler] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.get_streamed_result_of_chatgpt(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

    def get_streamed_result_of_chatgpt(self, prompt):
        prev_text = ""
        for data in self.chatbot.ask(prompt):
            prev_text = data["message"]
        return prev_text
    
    def init(self) -> None:
        """Initialize the LLM."""
        self.chatbot = Chatbot(
            config={
                "access_token":self.access_token,
                "model": self.model,
            },
        )