from phi.run.response import RunResponse

from agents import Agents

if __name__ == '__main__':
    agents = Agents().agent_team()
    # run: RunResponse = agents.run(" what happened in india in 1996 write an article from the searched content"
    #                               " about it")
    prompt="""
    Whats the current market trend and based on the market trend suggest best stocks to invest in india small cap  and give target prize
    give in tabular manner like this sno , stock name , cmp,target prize reason to buy 
    """
    agents.print_response(prompt,stream=True)
