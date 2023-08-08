# LangChain Experiment

Embark on a journey with LangChain, a next-generation platform that leverages the power of language models to build cutting-edge applications. Dive into the core components that make it all possible and see how you can get started today!

## Building Blocks of LangChain

LangChain's powerful framework is built upon three foundational elements that come together in the LLMChain:

1. **LLM (Language Logic Model):** At the heart of LangChain, the language model serves as the core reasoning engine. Familiarize yourself with various types of language models and learn how to manipulate them to suit your needs.
2. **Prompt Templates:** Like a conductor to an orchestra, these templates guide the language model, controlling what it outputs. Understanding prompt construction and the strategies behind them is essential.
3. **Output Parsers:** Transforming the raw response from the LLM into something more tangible, output parsers enable smooth downstream integration of the generated content.

## LangChain Communication Objects

- **HumanMessage:** A message originating from a user.
- **AIMessage:** A message generated by an AI or assistant.
- **SystemMessage:** A message provided by the system.
- **FunctionMessage:** A message resulting from a function call.

## Installation Guide

Get started with LangChain by following these installation steps:

```zsh
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## How to Run / Execute

### Experiment 0 - Hello LangChain

```zsh
source env/bin/activate
python3 experiment_0.py
```

### Experiment 1 - API Chains

```zsh
source env/bin/activate
python3 experiment_1_api_chains.py
```

### Experiment 2 - HTTP to GPT

```zsh
source env/bin/activate
python3 experiment_2_http_to_gpt.py
```

### Update Packages

```zsh
pip3 install -r requirements.txt --upgrade
```

## Documentation and References

- Explore deeper with the [LangChain documentation](https://python.langchain.com/docs) and become a LangChain maestro!
- [Navigate the Essentials of ChatGPT](https://elmerthomas.vercel.app/getting-started/openai/chatgpt).

## ToDo List

- [OpenAI Agents](https://python.langchain.com/docs/integrations/toolkits/openapi.html)
- [Question Answering](https://python.langchain.com/docs/use_cases/question_answering)
- [Model I/O & Prompts](https://python.langchain.com/docs/modules/model_io/prompts)
- [Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers)
- [Model I/O](https://python.langchain.com/docs/modules/model_io)
- [Modules](https://python.langchain.com/docs/modules)
- [Guides](https://python.langchain.com/docs/guides)
- [Use Cases](https://python.langchain.com/docs/use_cases)
- [OpenAPI Integrations](https://python.langchain.com/docs/integrations/toolkits/openapi.html)
