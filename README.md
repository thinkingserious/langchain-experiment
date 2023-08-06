# Langchain Experiment

Embark on a journey with Langchain, a next-generation platform that leverages the power of language models to build cutting-edge applications. Dive into the core components that make it all possible and see how you can get started today!

## Building Blocks of Langchain

LangChain's powerful framework is built upon three foundational elements that come together in the LLMChain:

1. **LLM (Language Logic Model):** At the heart of LangChain, the language model serves as the core reasoning engine. Familiarize yourself with various types of language models and learn how to manipulate them to suit your needs.
2. **Prompt Templates:** Like a conductor to an orchestra, these templates guide the language model, controlling what it outputs. Understanding prompt construction and the strategies behind them is essential.
3. **Output Parsers:** Transforming the raw response from the LLM into something more tangible, output parsers enable smooth downstream integration of the generated content.

## Langchain Communication Objects

- **HumanMessage:** A message originating from a user.
- **AIMessage:** A message generated by an AI or assistant.
- **SystemMessage:** A message provided by the system.
- **FunctionMessage:** A message resulting from a function call.

## Installation Guide

Get started with Langchain by following these installation steps:

```zsh
python3 -m venv env
source env/bin/activate
pip3 install langchain openai
pip3 install --upgrade pip
pip3 freeze > requirements.txt
export OPENAI_API_KEY="..."
```

## How to Run / Execute

Once installed, execute your experiment with the following commands:

```zsh
source env/bin/activate
export OPENAI_API_KEY="..."
python3 experiment.py
```

## Documentation and References

Explore deeper with the [LangChain documentation](https://python.langchain.com/docs) and become a LangChain maestro!

## ToDo List

- Enhance your understanding of Langchain through their comprehensive documentation:
    - [Model I/O & Prompts](https://python.langchain.com/docs/modules/model_io/prompts)
    - [Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers)
    - [Model I/O](https://python.langchain.com/docs/modules/model_io)
    - [Modules](https://python.langchain.com/docs/modules)
    - [Guides](https://python.langchain.com/docs/guides)
    - [Use Cases](https://python.langchain.com/docs/use_cases)
    - [Question Answering](https://python.langchain.com/docs/use_cases/question_answering)
    - [APIs](https://python.langchain.com/docs/use_cases/apis)
    - [API Documentation](https://python.langchain.com/docs/use_cases/apis/api.html)
    - [OpenAPI Integrations](https://python.langchain.com/docs/integrations/toolkits/openapi.html)