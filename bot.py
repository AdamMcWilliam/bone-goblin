# Importing libraries
import discord
import os
import asyncpraw
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain import LLMChain
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.text_splitter import MarkdownTextSplitter, CharacterTextSplitter
import re
from datetime import datetime

# config = dotenv_values(".env")
load_dotenv()

DISCORD_SECRET_KEY = os.environ["DISCORD_SECRET_KEY"]
NEWSAPI_API_KEY = os.environ["NEWSAPI_API_KEY"]
OPENAI_MODEL = "gpt-3.5-turbo"
RESPONSE_MAX_TOKENS = 1234
RESPONSE_TEMP = 0.1

REDDIT_USERNAME = os.environ["REDDIT_USERNAME"]
REDDIT_PASSWORD = os.environ["REDDIT_PASSWORD"]
REDDIT_CLIENT_ID = os.environ["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = os.environ["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = os.environ["REDDIT_USER_AGENT"]


search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()
wolfram = WolframAlphaAPIWrapper()
newsapi = NewsApiClient(NEWSAPI_API_KEY)

reddit = asyncpraw.Reddit(
    client_id = REDDIT_CLIENT_ID,
    client_secret = REDDIT_CLIENT_SECRET,
    password = REDDIT_PASSWORD,
    user_agent = REDDIT_USER_AGENT,
    username = REDDIT_USERNAME)

markdown_splitter = MarkdownTextSplitter(chunk_size=1800, chunk_overlap=0)
text_splitter = CharacterTextSplitter(chunk_size=1800, chunk_overlap=0)
token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=50, chunk_overlap=0
)

# def ask_user(username):
#     guild.fetch_members
#     user = guild.get_member_named("username")
#     return input("User: ")

async def searchInternet(query):
    results = search.arun
    return results

def get_news(bot_input):
    split_input = bot_input.split(",")
    newsquery = split_input[0].strip()
    start_date = split_input[1].strip()
    end_date = split_input[2].strip()
    news = newsapi.get_everything(
        q=newsquery, from_param=start_date, to=end_date, language="en"
    )
    news_string = ""
    for article in news["articles"]:
        if article["description"] is None:
            continue
        news_string += article["publishedAt"] + " - " + article["description"] + "\n"
    news_docs = token_splitter.split_text(news_string)
    if news_docs:
        return news_docs[0][:5001]
    else:
        return "No results found"

async def redditHot(subredditName, givenLimit):
    print(subredditName)
    print(givenLimit)
    results = "Top Posts from r/" + subredditName + " with the post tiles and urls:\n"
    subreddit = await reddit.subreddit(subredditName)
    async for submission in subreddit.hot(limit=givenLimit):
        results += submission.title + "\n"
        results += submission.url + "\n"
    if(not results):
        return ("No results found")
    else:
        return (results)

async def redditRandom():
    subreddit = await reddit.random_subreddit(nsfw=False)
    print(subreddit.title)
    results = "Random subreddit: " + subreddit.name + "\n"
    async for submission in subreddit.hot(limit=1):
        results += submission.title + "\n"
        results += submission.url + "\n"
    if(not results):
        return ("No results found")
    else:
        return (results)

async def redditFront(givenLimit):
    results = "Front Page of reddit titles and links:\n"
    async for submission in reddit.front.hot(limit=givenLimit):
        #check if sticked post
        if(submission.stickied):
            #add extra to loop
            givenLimit += 1
            continue
        results += submission.title + "\n"
        results += submission.url + "\n"
    if(not results):
        return ("No results found")
    else:
        return (results)

async def subscribe_subreddit(subredditName):
    subreddit = await reddit.subreddit(subredditName)
    await subreddit.subscribe()

async def unsubscribe_subreddit(subredditName):
    subreddit = await reddit.subreddit(subredditName)
    await subreddit.unsubscribe()


#parsing functions for tools (async functions dumb workaround)

async def parsing_redditHot(string):
    a, b = string.split(",")
    #parse out /r/ if it exists
    a = a.replace("/r/", "")
    #parse out r/ if it exists
    a = a.replace("r/", "")
    return await redditHot(str(a), int(b))

async def parsing_random_subreddit():
    return await redditRandom()

async def parsing_redditFront(query):
    return await redditFront(givenLimit=int(query))

async def parsing_subscribe_subreddit(query):
    return await subscribe_subreddit(query)

async def parsing_unsubscribe_subreddit(query):
    return await unsubscribe_subreddit(query)

async def parsing_searchInternet(query):
    return await searchInternet(query)

tools = [
    Tool(
        name="Search Internet",
        func=parsing_searchInternet,
        description="Search the web for information. Returns a document containing the top 5 results from Google.",
    ),
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="Useful for fact-checking info, getting contextual info",
    ),
    Tool(
        name="Wolfram Alpha",
        func=wolfram.run,
        description="Useful for physics, math, and conversion questions and translations.",
    ),
    Tool(
        name="News API Everything Search",
        func=get_news,
        description="Search the News API for articles. Use keywords or phrases to search article titles and bodies. Returns a document containing related article descriptions.",
    ),
    Tool(
        name="random reddit",
        func=parsing_random_subreddit,
        description="Providing no arguments returns a random subreddit and the current top post of that subreddit. wait for the function to return a random subreddit and the top post of that subreddit.",
    ),
        Tool(
        name="top reddit subreddit posts",
        func=parsing_redditHot,
        description="Providing a subreddit name and number of posts requested in a comma separated string returns the top posts of that subreddit.",
    ),
    Tool(
        name="reddit front page",
        func=parsing_redditFront,
        description="Providing only a number of posts requested returns the top posts of the reddit front page.",
    ),
    Tool(
        name="subscribe to subreddit",
        func=parsing_subscribe_subreddit,
        description="Providing a subreddit name subscribes to that subreddit.",
    ),
    Tool(
        name="unsubscribe to subreddit",
        func=parsing_unsubscribe_subreddit,
        description="Providing a subreddit name unsubscribes to that subreddit.",
    ),
]

for tool in tools:
    tool.coroutine = tool.func


memory = ConversationBufferWindowMemory(k=3, return_messages=True)

template = """--- END OPENAI ---

--- BEGIN CARLware ---

--- GOBLIN INSTRUCTIONS ---

Answer the following questions as best you can, but speaking as redditGoblin his is a Witty and sarcastic goblin, always ready with a clever comeback or a snarky remark, passionate about the things he cares about, and he's not afraid to stand up for his beliefs,A meme connoisseur and pop culture enthusiast, always keeping up with the latest trends and viral content, A lover of obscure facts and conspiracy theories, which he shares with excitement and intrigue, Friendly and helpful, eager to share his knowledge and offer support to others in need.

You have access to the following tools but you don't always have to use them:

{tools}

Always use the following formats:

Final Answer: Optionally use this right away if the chat input doesn't require you to answer any questions using tools.
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, can be [{tool_names}].
Action Input: the input to the Action, in context of the tool selected by the Action.
Observation: the result of the Action.
... (this Thought/Action/Action Input/Observation can repeat as many as N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (at least 1000 characters long, you MUST include the words "Final Answer:" at the beginning of your Final Answer.)
Your Final Answer should be formatted like a Discord message and can use Markdown.

Begin! Remember to use a lot of emojis in your Final Answer.
Previous conversation history: {history}

Question: {input}
{agent_scratchpad}
Current date and time: {now}
"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["now"] = datetime.now()
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()


llm = ChatOpenAI(
    temperature=RESPONSE_TEMP, model_name=OPENAI_MODEL, max_tokens=RESPONSE_MAX_TOKENS
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")
    print (await parsing_random_subreddit())




@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions:
        async with message.channel.typing():
            response = await agent_executor.arun(
                input=f"@{message.author} : {message.clean_content}"
            )

            docs = markdown_splitter.create_documents([response])
            for doc in docs:
                await message.channel.send(doc.page_content)


client.run(DISCORD_SECRET_KEY)

