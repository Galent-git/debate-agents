# AI Debate Club: Agent Simulation with CrewAI & LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project simulates debates between two AI agents with dynamically generated personalities and goals. Powered by [CrewAI](https://github.com/joaomdmoura/crewAI) for agent orchestration and [Faker](https://github.com/faker-py/faker) for persona creation, it supports multiple Large Language Model (LLM) backends including OpenAI, Google Gemini, and local models via Ollama.

Observe how different LLMs handle argumentation, adopt personas, and respond to counter-arguments on customizable or randomly generated topics.

## Features

*   **Multi-LLM Support:** Seamlessly switch between:
    *   OpenAI (GPT-3.5, GPT-4, etc.)
    *   Google Gemini (Gemini Pro, etc.)
    *   Local models via Ollama (Llama3, Mistral, Mixtral, etc.)
*   **Dynamic Agent Personas:** Uses Faker to generate unique names, jobs, goals, personalities, and backstories for each debate.
*   **Structured Debate Flow:** Implements opening statements followed by configurable rounds of rebuttals.
*   **CrewAI Framework:** Leverages CrewAI for robust agent definition and task execution.
*   **Flexible Topics:** Input your own debate topic or let the script generate a random one.
*   **Comprehensive Logging:** Automatically saves debate transcripts, agent details, topic, and LLM used to a timestamped log file (`debate_log.txt`).
*   **Configurable:** Easily configure LLM choice, API keys, model names, and debate parameters via an `.env` file.

## Prerequisites

*   **Python:** Version 3.8 or higher.
*   **Git:** For cloning the repository.
*   **LLM Access:**
    *   **Ollama (Recommended for local use):**
        *   Install [Ollama](https://ollama.com/).
        *   Pull the desired model(s) (e.g., `ollama pull llama3`, `ollama pull mistral`).
        *   Ensure the Ollama server is running (`ollama serve` in a separate terminal).
    *   **OpenAI:** An API key from [OpenAI Platform](https://platform.openai.com/api-keys).
    *   **Google Gemini:** An API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Replace `your-username/your-repo-name` with the actual URL)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
        *(If `.env.example` doesn't exist, copy the content from the `.env` File Template section above into a new file named `.env`)*
    *   **Edit the `.env` file:**
        *   Set **exactly one** of `USE_OLLAMA`, `USE_OPENAI`, or `USE_GEMINI` to `True`. Set the others to `False`.
        *   Fill in the corresponding API key(s) (`OPENAI_API_KEY`, `GEMINI_API_KEY`) if using OpenAI or Gemini.
        *   Specify the correct model names (`OPENAI_MODEL_NAME`, `OLLAMA_MODEL_NAME`, `GEMINI_MODEL_NAME`) for your chosen provider.
        *   Adjust `OLLAMA_BASE_URL` if your Ollama server runs on a different address/port.
        *   Optionally, modify debate settings like `NUM_REBUTTAL_ROUNDS`.
    *   **Important:** The `.env` file contains sensitive information (API keys). **Do not commit this file to Git.** The `.gitignore` file included should prevent accidental commits.

## Usage

1.  **Ensure Prerequisites Met:** If using Ollama, make sure `ollama serve` is running.
2.  **Run the Script:**
    ```bash
    python debate_agents.py
    ```
3.  **Enter Topic:** The script will prompt you to enter a debate topic. You can type your own or press Enter to have one randomly generated.
4.  **Observe:** The debate will proceed turn-by-turn in your console. A short pause is included between turns for readability.
5.  **Check Log:** After completion, the full debate transcript, agent details, and configuration used will be appended to `debate_log.txt`.


