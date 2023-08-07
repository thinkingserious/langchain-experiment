from langchain.chains.api import open_meteo_docs, tmdb_docs, podcast_docs
from langchain.chains import APIChain
from langchain.llms import OpenAI
from utils import print_separator
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

print_separator("STARTING EXPERIMENT - WEATHER")

llm = OpenAI(temperature=0)
chain_new = APIChain.from_llm_and_api_docs(
    llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=False)
print(chain_new.run(
    'What is the weather like right now in Munich, Germany in degrees Fahrenheit?'))
# >> The current temperature in Munich, Germany is 54.8°F.

print_separator("STARTING EXPERIMENT - MOVIES")

headers = {"Authorization": f"Bearer {os.getenv('TMDB_BEARER_TOKEN')}"}
chain = APIChain.from_llm_and_api_docs(
    llm, tmdb_docs.TMDB_DOCS, headers=headers, verbose=False)
print(chain.run("Search for 'Pulp Fiction'"))
# >> The response from the API contains 7 movies related to the search query 'Pulp Fiction'.
#    The most popular movie is 'Pulp Fiction' (1994) starring John Travolta and Uma Thurman.
#    Other movies include 'Pulp Fiction: The Facts' (2002), 'Stealing Pulp Fiction' (2020),
#    'Making of Pulp Fiction' (2000), 'Pulp Fiction Art' (2005), 'Stealing Pulp Fiction' (2020)
#    and 'Pulp Fiction: The Golden Age of Storytelling' (2009).

print_separator("STARTING EXPERIMENT - PODCASTS")

listen_api_key = os.getenv('LISTEN_API_KEY')
llm = OpenAI(temperature=0)
headers = {"X-ListenAPI-Key": listen_api_key}
print(listen_api_key)
chain = APIChain.from_llm_and_api_docs(
    llm, podcast_docs.PODCAST_DOCS, headers=headers, verbose=False)
print(chain.run(
    "Search for 'star wars' podcast episodes, audio length is more than 30 minutes, return 1 result and provide the URL."))
# >> The API call returned 1 result for a podcast episode about Star Wars announcing 3 movies at Star Wars Celebration.
#    The episode has an audio length of 4242 seconds (70 minutes) and the URL for the episode is
#    https://www.listennotes.com/e/p/562f76277ddd471dbba61a878363bc62/.
