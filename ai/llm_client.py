from typing import List, Dict, Optional, Iterator, AsyncIterator

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from utils import get_config, logger, AIError


class LLMClient:

    def __init__(self, streaming: bool = False):
        self.config = get_config()
        self.streaming = streaming

        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            temperature=0.7,
            max_tokens=2000,
            callbacks=callbacks,
            streaming=streaming
        )

        logger.info(f"Initialized LLM client with model: {self.config.openai_model}")

    def get_llm(self) -> BaseLanguageModel:
        return self.llm

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            history: Optional[List[BaseMessage]] = None,
            stream: bool = False,
            **kwargs
    ) -> str:
        try:
            # messages = []
            #
            # # Add system message if provided
            # if system_prompt:
            #     messages.append(SystemMessage(content=system_prompt))
            #
            # # Add conversation history
            # if history:
            #     messages.extend(history)
            #
            # # Add current user message
            # messages.append(HumanMessage(content=prompt))
            #
            # # Generate response
            # response = self.llm.invoke(messages, **kwargs)
            #
            # logger.info(f"Generated response with {len(response.content)} characters")
            # return response.content

            messages = self._build_messages(prompt, system_prompt, history)

            if stream:
                return self._stream_response(messages, **kwargs)
            else:
                return self._generate_response(messages, **kwargs)

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise AIError(f"LLM generation failed: {str(e)}")

    async def agenerate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            history: Optional[List[BaseMessage]] = None,
            stream: bool = False,
            **kwargs
    ) -> str:
        try:
            # messages = []
            #
            # if system_prompt:
            #     messages.append(SystemMessage(content=system_prompt))
            #
            # if history:
            #     messages.extend(history)
            #
            # messages.append(HumanMessage(content=prompt))
            #
            # # Async generation
            # response = await self.llm.ainvoke(messages, **kwargs)
            #
            # logger.info(f"Generated async response with {len(response.content)} characters")
            # return response.content

            messages = self._build_messages(prompt, system_prompt, history)

            if stream:
                return self._astream_response(messages, **kwargs)
            else:
                return await self._agenerate_response(messages, **kwargs)

        except Exception as e:
            logger.error(f"Async LLM generation failed: {str(e)}")
            raise AIError(f"Async LLM generation failed: {str(e)}")

    def chat(
            self,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None,
            stream: bool = False,
            **kwargs
    ) -> str:
        try:
            # Convert message dicts to LangChain messages
            # lc_messages = []
            #
            # if system_prompt:
            #     lc_messages.append(SystemMessage(content=system_prompt))
            #
            # for msg in messages:
            #     role = msg.get('role', 'user')
            #     content = msg.get('content', '')
            #
            #     if role == 'system':
            #         lc_messages.append(SystemMessage(content=content))
            #     elif role == 'assistant':
            #         lc_messages.append(AIMessage(content=content))
            #     else:  # user
            #         lc_messages.append(HumanMessage(content=content))
            #
            # # Generate response
            # response = self.llm.invoke(lc_messages, **kwargs)
            #
            # return response.content

            lc_messages = self._convert_messages(messages, system_prompt)

            if stream:
                return self._stream_response(lc_messages, **kwargs)
            else:
                return self._generate_response(lc_messages, **kwargs)

        except Exception as e:
            logger.error(f"Chat generation failed: {str(e)}")
            raise AIError(f"Chat generation failed: {str(e)}")

    def set_model(self, model_name: str):
        self.llm.model_name = model_name
        logger.info(f"Changed LLM model to: {model_name}")

    def set_temperature(self, temperature: float):
        self.llm.temperature = max(0.0, min(1.0, temperature))
        logger.info(f"Set temperature to: {self.llm.temperature}")

    def set_max_tokens(self, max_tokens: int):
        self.llm.max_tokens = max_tokens
        logger.info(f"Set max_tokens to: {max_tokens}")

    def _build_messages(
            self,
            prompt: str,
            system_prompt: Optional[str],
            history: Optional[List[BaseMessage]]
    ) -> List[BaseMessage]:
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(HumanMessage(content=prompt))
        return messages

    def _convert_messages(
            self,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str]
    ) -> List[BaseMessage]:
        lc_messages = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            elif role == 'assistant':
                lc_messages.append(AIMessage(content=content))
            else:  # user
                lc_messages.append(HumanMessage(content=content))

        return lc_messages

    def _generate_response(self, messages: List[BaseMessage], **kwargs) -> str:
        response = self.llm.invoke(messages, **kwargs)
        logger.info(f"Generated response with {len(response.content)} characters")
        return response.content

    def _stream_response(self, messages: List[BaseMessage], **kwargs) -> Iterator[str]:
        try:
            total_chars = 0

            for chunk in self.llm.stream(messages, **kwargs):
                if chunk.content:
                    total_chars += len(chunk.content)
                    yield chunk.content

            logger.info(f"Streaming completed: {total_chars} characters")

        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            yield f"\n\n[Error: {str(e)}]"

    async def _agenerate_response(self, messages: List[BaseMessage], **kwargs) -> str:
        response = await self.llm.ainvoke(messages, **kwargs)
        logger.info(f"Generated async response with {len(response.content)} characters")
        return response.content

    async def _astream_response(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[str]:
        try:
            total_chars = 0

            async for chunk in self.llm.astream(messages, **kwargs):
                if chunk.content:
                    total_chars += len(chunk.content)
                    yield chunk.content

            logger.info(f"Async streaming completed: {total_chars} characters")

        except Exception as e:
            logger.error(f"Async streaming failed: {str(e)}")
            yield f"\n\n[Error: {str(e)}]"
