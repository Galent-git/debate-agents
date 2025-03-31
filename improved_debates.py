import os
import random
import time
from datetime import datetime

import requests
from faker import Faker
# Required libraries for Langchain, CrewAI, and LLMs
from langchain_community.llms import Ollama, OpenAI
# Placeholder for potential Google Gemini integration
# from langchain_google_genai import ChatGoogleGenerativeAI # Example import
from crewai import Agent

# === Configuration ===

# --- LLM Selection ---
# Set exactly one of these to True
USE_OPENAI = False
USE_OLLAMA = True
USE_GEMINI = False  # Placeholder for future integration

# --- API Keys & Model Names ---
# Provide your OpenAI API key if USE_OPENAI is True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
OPENAI_MODEL_NAME = "gpt-4"  # Or "gpt-3.5-turbo", etc.

# Set your Ollama model name if USE_OLLAMA is True (e.g., "llama3", "mistral", "llama2")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3") # Default to llama3 if env var not set
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # For remote Ollama

# Provide your Google API key if USE_GEMINI is True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-google-api-key-here")
GEMINI_MODEL_NAME = "gemini-pro" # Or other suitable Gemini model

# --- Debate Settings ---
NUM_REBUTTAL_ROUNDS = 2  # Number of rebuttal rounds after opening statements
DEBATE_LOG_FILE = "debate_log.txt"
SLEEP_TIME_BETWEEN_TURNS = 1 # Seconds to pause between agent turns

# --- Initialize Faker ---
fake = Faker()

# === Helper Functions ===

def check_ollama_alive(url: str) -> bool:
    """
    Checks if the Ollama server is running and accessible.

    Args:
        url (str): The base URL of the Ollama server.

    Returns:
        bool: True if the server responds with status 200, False otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print(f"✅ Ollama server found at {url}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Ollama server not found at {url}. Is it running?")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to Ollama server at {url}: {e}")
        return False

def get_llm():
    """
    Initializes and returns the selected LangChain LLM instance based on configuration.

    Handles selection between OpenAI, Ollama, and a placeholder for Google Gemini.
    Exits if the selected Ollama server is not reachable.

    Returns:
        Union[OpenAI, Ollama, None]: The initialized LangChain LLM instance,
                                      or None if Gemini is selected (as it's a placeholder).
                                      Exits script if selected provider fails.
    """
    if USE_OPENAI:
        if OPENAI_API_KEY == "your-openai-api-key-here" or not OPENAI_API_KEY:
             print("❗ WARNING: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable or edit the script.")
             # Decide if you want to exit or proceed with a potential error later
             # exit(1) # Uncomment to exit if key is missing
        print(f"⚙️ Using OpenAI model: {OPENAI_MODEL_NAME}")
        return OpenAI(model=OPENAI_MODEL_NAME, temperature=0.7, api_key=OPENAI_API_KEY)

    elif USE_OLLAMA:
        if not check_ollama_alive(OLLAMA_BASE_URL):
             print("❗ Please ensure the Ollama server is running and accessible.")
             exit(1)
        print(f"⚙️ Using Ollama model: {OLLAMA_MODEL_NAME} from {OLLAMA_BASE_URL}")
        return Ollama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)

    elif USE_GEMINI:
        if GEMINI_API_KEY == "your-google-api-key-here" or not GEMINI_API_KEY:
             print("❗ WARNING: Google API key is not set. Please set the GEMINI_API_KEY environment variable or edit the script.")
             # exit(1) # Uncomment to exit if key is missing
        print(f"⚙️ Using Google Gemini model: {GEMINI_MODEL_NAME} (Placeholder)")
        # --- Google Gemini Integration Placeholder ---
        # 1. Ensure you have the necessary package:
        #    pip install langchain-google-genai google-generativeai
        # 2. Import the class:
        #    from langchain_google_genai import ChatGoogleGenerativeAI
        # 3. Instantiate the LLM:
        #    return ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0.7)
        # --- End Placeholder ---
        print("🚧 Google Gemini integration is currently a placeholder. Add implementation in get_llm().")
        # Return None for now, or raise NotImplementedError
        # return None
        raise NotImplementedError("Google Gemini integration not yet implemented in this script.")

    else:
        print("❌ ERROR: No LLM provider selected. Set USE_OPENAI, USE_OLLAMA, or USE_GEMINI to True.")
        exit(1)


def generate_debater_identity() -> dict:
    """
    Generates a dictionary containing a fictional debater's identity using Faker.

    Returns:
        dict: A dictionary with keys 'name', 'job', 'personality', 'goal', 'backstory'.
    """
   name = fake.name()
    job = fake.job()
    personality = fake.sentence(nb_words=random.randint(8, 15))
    goal = fake.catch_phrase()
    # Corrected backstory generation - uses the *generated* name
    backstory = (
        f"Known for their {fake.bs()} approach, {name} believes deeply in "
        f"{fake.company_suffix()} principles. {fake.paragraph(nb_sentences=2)}"
    )
    # Return the collected dictionary
    return {
        "name": name,
        "job": job,
        "personality": personality,
        "goal": goal,
        "backstory": backstory
    }

def run_debate_turn(agent: Agent, agent_identity: dict, opponent_statement: str, topic: str, turn_description: str) -> str:
    """
    Runs a single turn for an agent in the debate.

    Args:
        agent (Agent): The CrewAI agent instance taking the turn.
        agent_identity (dict): The identity dictionary of the agent.
        opponent_statement (str): The previous statement made by the opponent.
        topic (str): The main debate topic.
        turn_description (str): A description of the current turn (e.g., "Opening Statement", "Rebuttal 1").

    Returns:
        str: The statement generated by the agent for this turn.
    """
    if opponent_statement:
        prompt = (
            f"Debate Topic: '{topic}'\n"
            f"Your Role: {agent_identity['job']} ({agent_identity['personality']})\n"
            f"Your Goal: {agent_identity['goal']}\n"
            f"Your opponent, {opponent_statement.split(':')[0] if ':' in opponent_statement else 'your opponent'}, just said: '{opponent_statement.split(':', 1)[1].strip() if ':' in opponent_statement else opponent_statement}'\n"
            f"Now, deliver your {turn_description}. Make a clear point and directly address their argument if possible."
        )
    else:
        # Opening statement
        prompt = (
             f"Debate Topic: '{topic}'\n"
             f"Your Role: {agent_identity['job']} ({agent_identity['personality']})\n"
             f"Your Goal: {agent_identity['goal']}\n"
             f"Deliver your powerful {turn_description}. State your main position clearly."
        )

    print(f"\n--- {agent_identity['name']} ({turn_description}) ---")
    statement = agent.run(prompt)
    print(f"{agent_identity['name']}:\n{statement}\n")
    time.sleep(SLEEP_TIME_BETWEEN_TURNS)
    return f"{agent_identity['name']}: {statement}" # Prepend name for history

def log_debate(filename: str, topic: str, agent_1_id: dict, agent_2_id: dict, history: list):
    """
    Logs the details and transcript of the debate to a file.

    Args:
        filename (str): The path to the log file.
        topic (str): The debate topic.
        agent_1_id (dict): Identity of the first agent.
        agent_2_id (dict): Identity of the second agent.
        history (list): A list of strings, each representing a turn in the debate.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = "\n".join(history) # Each item in history already has the name prepended

    log_content = f"""
===== Debate Log: {timestamp} =====
Topic: {topic}

--- Agent 1 ---
Name: {agent_1_id['name']}
Role: {agent_1_id['job']}
Goal: {agent_1_id['goal']}
Personality: {agent_1_id['personality']}
Backstory: {agent_1_id['backstory']}

--- Agent 2 ---
Name: {agent_2_id['name']}
Role: {agent_2_id['job']}
Goal: {agent_2_id['goal']}
Personality: {agent_2_id['personality']}
Backstory: {agent_2_id['backstory']}

--- Debate Transcript ---
{summary}
================ End Log ================

"""
    try:
        with open(filename, "a", encoding="utf-8") as log_file:
            log_file.write(log_content)
        print(f"📄 Debate logged to {filename}")
    except IOError as e:
        print(f"❌ ERROR: Could not write to log file {filename}: {e}")

# === Main Execution ===

if __name__ == "__main__":

    # --- Initialize LLM ---
    llm = get_llm()
    if llm is None and USE_GEMINI:
         print("Exiting because Gemini integration is not complete.")
         exit(1)
    elif llm is None:
         print("❌ ERROR: Failed to initialize LLM. Exiting.")
         exit(1)


    # --- Define Debate Topic ---
    user_topic = input("Enter a debate topic (or press Enter for a random one): ").strip()
    # Use a slightly more engaging random topic generator if none provided
    topic = user_topic if user_topic else f"Should {fake.company()} be allowed to {fake.bs()}?"
    print(f"\n💬 Debate Topic: {topic}\n")

    # --- Generate Debaters ---
    identity_1 = generate_debater_identity()
    identity_2 = generate_debater_identity()

    print(f"👤 Debater 1: {identity_1['name']} ({identity_1['job']})")
    print(f"   Goal: {identity_1['goal']}")
    print(f"👤 Debater 2: {identity_2['name']} ({identity_2['job']})")
    print(f"   Goal: {identity_2['goal']}")

    # --- Create CrewAI Agents ---
    agent_1 = Agent(
        role=identity_1['job'],
        goal=f"Win the debate about '{topic}' by arguing persuasively based on your persona. Your ultimate goal is: {identity_1['goal']}",
        backstory=(
            f"You are {identity_1['name']}, a {identity_1['job']}. "
            f"{identity_1['personality']}. {identity_1['backstory']}"
        ),
        llm=llm,
        verbose=False # Set to True for more detailed agent logging
    )

    agent_2 = Agent(
        role=identity_2['job'],
        goal=f"Win the debate about '{topic}' by arguing persuasively based on your persona. Your ultimate goal is: {identity_2['goal']}",
        backstory=(
            f"You are {identity_2['name']}, a {identity_2['job']}. "
            f"{identity_2['personality']}. {identity_2['backstory']}"
        ),
        llm=llm,
        verbose=False
    )

    # --- Run the Debate ---
    debate_history = []
    last_statement = ""

    print("\n=== Debate Start ===")

    # Opening Statements
    statement_1 = run_debate_turn(agent_1, identity_1, None, topic, "Opening Statement")
    debate_history.append(statement_1)
    last_statement = statement_1

    statement_2 = run_debate_turn(agent_2, identity_2, last_statement, topic, "Opening Statement / Rebuttal")
    debate_history.append(statement_2)
    last_statement = statement_2

    # Rebuttal Rounds
    for i in range(NUM_REBUTTAL_ROUNDS):
        round_num = i + 1
        print(f"\n=== Rebuttal Round {round_num} ===")

        # Agent 1's turn
        rebuttal_1 = run_debate_turn(agent_1, identity_1, last_statement, topic, f"Rebuttal {round_num}")
        debate_history.append(rebuttal_1)
        last_statement = rebuttal_1

        # Agent 2's turn
        rebuttal_2 = run_debate_turn(agent_2, identity_2, last_statement, topic, f"Rebuttal {round_num}")
        debate_history.append(rebuttal_2)
        last_statement = rebuttal_2

    # --- Display Final Summary ---
    print("\n=== Debate Concluded ===")
    # print("\n--- Full Transcript ---")
    # for turn in debate_history:
    #     print(turn) # Name is already included

    # --- Log Debate ---
    log_debate(DEBATE_LOG_FILE, topic, identity_1, identity_2, debate_history)

    print("\n✅ Debate finished and logged.")


