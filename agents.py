from phi.agent import Agent as Agent
from phi.model.openai import OpenAIChat
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.huggingface import HuggingFaceChat
from phi.agent import Agent, RunResponse
from phi.model.ollama import Ollama
from phi.model.nvidia import Nvidia
from phi.model.google import Gemini
from phi.tools.calculator import Calculator
from phi.tools.openbb_tools import OpenBBTools
from dotenv import load_dotenv
import os

from phi.tools.newspaper4k import Newspaper4k


class Agents:
    """"
    This module initializes the AI model agents
    """
    openai_model ="gpt-3.5-turbo"
    gemini_model ="gemini-1.5-flash"
    ollama_model ="llama3.2"
    hf_model ="meta-llama/Meta-Llama-3-8B-Instruct"
    
    def __init__(self):
        # Load env
        load_dotenv()
        # open ai chat 
        self.openai_chat=OpenAIChat(id=self.openai_model,
                                      api_key=os.getenv("OPENAI_API_KEY"))
        # Hugging face 
        self.huggingface=HuggingFaceChat(
        id=self.hf_model,
        api_key=os.getenv("HUGGING_FACE_API"),
        max_tokens=4096,
    )
        # Ollama Model
        self.model = Ollama(id=self.ollama_model)
        
        # Gemini model
        self.gemini =Gemini(id=self.gemini_model,
                            api_key=os.getenv("GEMINI_API_KEY"))
        
        
    def web_search_agent(self):
        agent =Agent(
            name="Web Agent",
            model=self.gemini,
            tools=[DuckDuckGo()],
            instructions=["Always include sources"],
            show_tool_calls=True,
            markdown=True,
        )
        return agent

    def openbb_agent(self):
        agent = OpenBBTools(provider="yfinance",
                            stock_price=True,
                           company_news=True ,
                           price_targets=True,
                           company_profile=True,
                           search_symbols=True,
                        #    model =self.gemini
                        )
        return agent

    def research_agent(self):
        agent = Agent(
            model=self.gemini,
            tools=[DuckDuckGo(search=True,news=True),
                   Newspaper4k(read_article=True,include_summary=True)],
            description="You are a senior NYT researcher writing an article on a topic.",
            instructions=[
                "For a given topic, search for the top 5 links.",
                "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
                "Analyse and prepare an NYT worthy article based on the information.",
            ],
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )
        return agent
    
    
    def calculator_agent(self):
        agent = Agent(
            model=self.gemini,
        tools=[
        Calculator(
            add=True,
            subtract=True,
            multiply=True,
            divide=True,
            exponentiate=True,
            factorial=True,
            is_prime=True,
            square_root=True,
            enable_all= True
        )
    ],
    show_tool_calls=True,
    markdown=True,
        )
        return agent; 







    def agent_team(self):
        agent_team = Agent(
            team=[self.web_search_agent(),
                  self.research_agent(),
                  self.calculator_agent(), 
                  self.openbb_agent()],
            
            
            instructions=["Always include sources",
                          "Use tables to display data"],
            show_tool_calls=True,
            markdown=True,
            model=self.gemini,
            description=
            """ research agent that can search the web, read the top links and write a report Web agent can get 
            information from web you can use web agent for getting realtime data.
            Use calculator agent to prform basic math operations
            Use openbb_agent to acces and analyze stocks and companies
            """
        )
        return agent_team






    
    
    
   
    
        
 
     
 
    