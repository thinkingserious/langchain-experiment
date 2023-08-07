from langchain.schema import BaseOutputParser

# Define a class to parse the output of an LLM call to a comma-separated list
class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

# Define a function to print a separator with a message
def print_separator(message):
    """Print a separator with a message."""
    separator = "=" * 40
    print(f"{separator}\n{message}\n{separator}")