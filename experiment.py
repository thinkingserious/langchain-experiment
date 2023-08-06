from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser

# Define a class to parse the output of an LLM call to a comma-separated list
class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

# Test the CommaSeparatedListOutputParser class
print(CommaSeparatedListOutputParser().parse("hi, bye"))

# Define a template for a chat prompt that translates input_language to output_language
template = "You are a helpful assistant that translates {input_language} to {output_language}."
# Create a system message prompt from the template
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# Define a template for a human message prompt
human_template = "{text}"
# Create a human message prompt from the template
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# Create a chat prompt from the system and human message prompts
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# Test the chat prompt by formatting it with input_language, output_language, and text
print(chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming."))

# Define a template for a chat prompt that generates comma-separated lists
template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
# Create a system message prompt from the template
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# Define a template for a human message prompt
human_template = "{text}"
# Create a human message prompt from the template
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# Create a chat prompt from the system and human message prompts
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# Create an LLMChain object that uses the ChatOpenAI model, the chat prompt, and the CommaSeparatedListOutputParser object
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser()
)
# Test the LLMChain object by generating a list of 5 colors
print(chain.run("colors"))
# >> ['red', 'blue', 'green', 'yellow', 'orange']