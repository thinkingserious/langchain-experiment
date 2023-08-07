from langchain.chains.api import open_meteo_docs
from langchain.chains.api import tmdb_docs
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
print(os.getenv('TMDB_BEARER_TOKEN'))

headers = {"Authorization": f"Bearer {os.getenv('TMDB_BEARER_TOKEN')}"}
chain = APIChain.from_llm_and_api_docs(llm, tmdb_docs.TMDB_DOCS, headers=headers, verbose=True)
print(chain.run("Search for 'Avatar'"))
