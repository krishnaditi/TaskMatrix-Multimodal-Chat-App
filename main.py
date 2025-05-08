import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.language_models.llms import LLM
from groq import Groq
from pydantic import PrivateAttr

# Load environment variables
_ = load_dotenv(find_dotenv(), override=True)

# Custom LangChain-compatible LLM wrapper using Groq client
class GrokLLM(LLM):
    _client: Groq = PrivateAttr()
    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()

    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._client = Groq(api_key=api_key)
        self._model = model
        self._temperature = temperature

    @property
    def _llm_type(self):
        return "grok"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=stop,
        )
        return response.choices[0].message.content

# Initialize Grok-backed LLM
llm = GrokLLM(api_key=os.getenv("GROQ_API_KEY"))

# Prompt template for summarization
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize: {text}"
)

# Create chain
summarize_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Wrapper function
def run_summary(text):
    return summarize_chain.run(text)

# Entry point
if __name__ == "__main__":
    text = input("Enter text to summarize: ")
    print(run_summary(text))