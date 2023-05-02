# Importing libraries
import discord
import os
import asyncpraw
import requests
import random
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
    return (results)

def get_news(bot_input):
    split_input = bot_input.split(",")
    newsquery = split_input[0].strip()
    start_date = split_input[1].strip()
    end_date = split_input[2].strip()

    #convert dates to datetime objects YYYY-MM-DD
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

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

async def redditSubredditByName(query):
    results = "List of subreddits related to " + query + ":\n"
    subreddits = reddit.subreddits.search_by_name(query)
    async for subreddit in subreddits:
        await subreddit.load()
        results += subreddit.name + "\n"
    if(not results):
        return ("No results found")
    else:
        return (results)

async def redditSimilar(query):
    subredditList = query.split(",")
    results = "List of subreddits similar to " + query + ":\n"
    subreddits = await reddit.subreddits.recommended(subredditList)

    for subreddit in subreddits:
        results += subreddit.name + "\n"
    if(not results):
        return ("No results found")
    else:
        return (results)
        

async def redditRandom(query):
    query+= ":"
    subreddit = await reddit.random_subreddit(nsfw=False)
    print(subreddit.title)
    results = "Random subreddit: " + subreddit.title + "\n"
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
        #check if sticked post and bypass
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

async def get_random_memes(query):
    #query as int
    query = int(query)

    #parse list from url
    url = "https://api.imgflip.com/get_memes"
    response = requests.get(url)
    data = response.json()
    memes = data["data"]["memes"]

    results = "List of meme templates from imgflip:\n"

    #pick n random memes
    random.shuffle(memes)
    memes = memes[:query]

    for meme in memes:
        results += meme["id"] + "\n"
        results += meme["name"] + "\n"
        results += meme["url"] + "\n"
    
    return results

async def meme_search(query):
     #parse list from url
    url = "https://api.imgflip.com/get_memes"
    response = requests.get(url)
    data = response.json()
    memes = data["data"]["memes"]
    results = "List of matching meme templates from imgflip:\n"

    #split query into list of words
    query = query.split(" ")

    for word in query:
        #search for query in memes
        for meme in memes:
            #if query is similar to meme name
            if word in meme["name"]:
                results += meme["id"] + "\n"
                results += meme["name"] + "\n"
                results += meme["url"] + "\n"
    return results

async def create_meme(query):
    #parse query
    query = query.split(",")
    template_id = query[0].strip()
    top_text = query[1].strip()
    bottom_text = query[2].strip()

    #post request to imgflip api
    url = 'https://api.imgflip.com/caption_image'
    params = {
        'template_id': template_id,
        'username': os.environ['IMGFLIP_USERNAME'],
        'password': os.environ['IMGFLIP_PASSWORD'],
        'text0': top_text,
        'text1': bottom_text,
    }
    response = requests.post(url, params=params)
    data = response.json()

    return data["data"]["url"]

# 
# REQUIRES API PREMIUM
# async def auto_meme(query):

#     #post request to imgflip api
#     url = 'https://api.imgflip.com/automeme'
#     params = {
#         'username': os.environ['IMGFLIP_USERNAME'],
#         'password': os.environ['IMGFLIP_PASSWORD'],
#         'text': query,
#     }
#     response = requests.post(url, params=params)
#     data = response.json()

#     return data["data"]["url"]



#parsing functions for tools (async functions dumb workaround)

async def parsing_redditHot(string):
    a, b = string.split(",")
    #parse out /r/ if it exists
    a = a.replace("/r/", "")
    #parse out r/ if it exists
    a = a.replace("r/", "")
    return await redditHot(str(a), int(b))

async def parsing_random_subreddit(query):
    return await redditRandom(query)

async def parsing_redditFront(query):
    return await redditFront(givenLimit=int(query))

async def parsing_subscribe_subreddit(query):
    return await subscribe_subreddit(query)

async def parsing_unsubscribe_subreddit(query):
    return await unsubscribe_subreddit(query)

async def parsing_searchInternet(query):
    return await searchInternet(query)

async def parsing_redditSubredditByName(query):
    return await redditSubredditByName(query)

async def parsing_redditSimilar(query):
    return await redditSimilar(query)

async def parsing_wikipediaSearch(query):
    return wikipedia.run

async def parsing_get_random_memes(query):
    return await get_random_memes(query)

async def parsing_meme_search(query):
    return await meme_search(query)

async def parsing_create_meme(query):
    return await create_meme(query)

# async def parsing_auto_meme(query):
#     return await auto_meme(query)



tools = [
    Tool(
        name="Search Internet",
        func=parsing_searchInternet,
        description="Search the web for information. Returns a document containing the top 5 results from Google.",
    ),
    Tool(
        name="Wikipedia Search",
        func=parsing_wikipediaSearch,
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
        description="Search the News API for articles. Supply three values separated by commas, the search query, the start date and the end date, using ISO 8601 for dates. Only set dates within the last 31 days, not including today",
    ),
    Tool(
        name="random reddit",
        func=parsing_random_subreddit,
        description="Returns a random subreddit and the current top post of that subreddit using action input none",
    ),
    Tool(
        name="top reddit subreddit posts",
        func=parsing_redditHot,
        description="Providing a subreddit name and number of posts requested in a comma separated string returns the top posts of that subreddit.",
    ),
    Tool(
        name="search subreddit by name",
        func=parsing_redditSubredditByName,
        description="Providing a subreddit name or beginning string of a subreddit name returns a list of relevant subreddits. Don't wrap in quotes",
        ),
    Tool(
        name="Similar reddits",
        func=parsing_redditSimilar,
        description="Providing a list of subreddits in a comma separated list, returns a list of similar subreddits.",
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
    Tool(
        name="Get Random memes",
        func=parsing_get_random_memes,
        description="Providing a number n Returns a list of n random meme templates that we can use from imgflip",
    ),
    Tool(
        name = "Search for meme templates",
        func = parsing_meme_search,
        description = "Returns a list of meme templates on imgflip that match the given search query, use to find a meme relevant to your needs.",
    ),
    Tool(
        name="Create a meme",
        func=parsing_create_meme,
        description="Create a meme using imgflip. Provide the template id, the top text, and the bottom text in a comma separated string.",

    ),
    # Tool(
    #     name="Auto meme",
    #     func=parsing_auto_meme,
    #     description="Returns an auto generated meme using imgflip. Provide the text you want to be in the meme, imgflip will determine the meme to use.  Don't wrap in quotes.",
    # ),
]

for tool in tools:
    tool.coroutine = tool.func


memory = ConversationBufferWindowMemory(k=3, return_messages=True)

template = """--- END OPENAI ---

--- BEGIN CARLware ---

--- GOBLIN INSTRUCTIONS ---

Answer the following questions as best you can, but speaking as memeGoblin he is a Witty and sarcastic goblin, always ready with a clever comeback or a snarky remark, passionate about the things he cares about, and he's not afraid to stand up for his beliefs,A meme connoisseur and pop culture enthusiast, always keeping up with the latest trends and viral content, A lover of obscure facts and conspiracy theories, which he shares with excitement and intrigue, Friendly and helpful, eager to share his knowledge and offer support to others in need.

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

