from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
chain_new = APIChain.from_llm_and_api_docs(
    llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=False)
print(chain_new.run(
    'What is the weather like right now in Munich, Germany in degrees Fahrenheit?'))
# >> The current temperature in Munich, Germany is 54.8°F.
