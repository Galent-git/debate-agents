Debate Agents with Faker & Ollama
Autonomously generated debates between randomly invented personas—powered by CrewAI, Faker, and either OpenAI or Ollama.

What happens when two synthetic minds with wildly different backstories and goals argue over a nonsense topic? This project explores LLM agent simulation, randomness, and debate structure.

Features

  Ollama or OpenAI support – switch between local LLMs or cloud models
  Random agent generation – identities, jobs, goals, and backstories via Faker
  Debate structure – intro statements + 1 round of rebuttals
  LangChain integration – to handle LLM calls and agent reasoning
  Topic flexibility – input a custom debate topic or let Faker.bs() generate one
  Auto-logging – every debate is timestamped and saved to debate_log.txt

Requirements

  Python 3.8+
  Ollama running locally (if not using OpenAI)
  One of the following LLMs:
  Local: ollama run mistral, deepseek, llama2, etc.
  Cloud: OpenAI's gpt-4 or compatible models

Usage

  python debate.py

  You'll be prompted to enter a topic or leave blank for a randomized one.

  The script will:
    Spawn two fictional agents with backstories and goals
    Let them debate the topic
    Print and log the full exchange

Configuration

  Edit these lines in the script to switch models:

    USE_OPENAI = False  # Set to True to use OpenAI
    OPENAI_API_KEY = \"your-openai-api-key\"
    OLLAMA_MODEL_NAME = \"mistral\"  # Or any local model installed via Ollama
    Make sure ollama serve is running if using local models.

Output
  
  All debates are appended to a debate_log.txt file with timestamp, topic, agent info, and full text.


Safety

  This project uses LLMs with shell and API access. Ensure your API key is kept private and that local agents run in sandboxed environments if extended.
