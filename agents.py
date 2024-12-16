from phi.agent import Agent as Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.huggingface import HuggingFaceChat
from phi.agent import Agent, RunResponse
from phi.model.ollama import Ollama
from phi.model.nvidia import Nvidia
from phi.model.google import Gemini
from dotenv import load_dotenv
import os

class Agents:
    openai_model ="gpt-3.5-turbo"
    gemini_model ="gemini-1.5-flash"
    ollama_model ="llama3.2"
    hf_model ="meta-llama/Meta-Llama-3-8B-Instruct"
    
    def __init__(self):
        # Load env
        load_dotenv()
        # open ai chat 
        self.openai_chat=OpenAIChat(id=self.__openai_model,
                                      api_key=os.getenv("OPENAI_API_KEY"))
        
        # Hugging face 
        self.huggingface=HuggingFaceChat(
        id=self.hf_model,
        api_key=os.getenv("HUGGING_FACE_API"),
        max_tokens=4096,
    )
        
        
        self.model = Ollama(id=self.ollama_model)
        
        self.gemini =Gemini(id=self.gemini_model,
                            api_key=os.getenv("GEMINI_API_KEY"))
        
        pass
    
    def web_agent(self,instructions=["Always include sources"]) ->Agent:
        agent = Agent(
            
    name="Web Agent",
    model=self.gemini,
    tools=[DuckDuckGo(search=True,news=True)],
    instructions=instructions,
    show_tool_calls=True,
    markdown=True)
        
        
        return agent
    
        
web = Agents().web_agent()
web.print_response("Whats happening in France?", stream=True) 
 
    