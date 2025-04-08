import os
import random
import time
from datetime import datetime
import sys

import requests
from faker import Faker
from dotenv import load_dotenv

# Langchain & CrewAI Imports
# Note: Adjust imports based on potential future changes in langchain/crewai versions
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain_google_genai import ChatGoogleGenerativeAI
    from crewai import Agent
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# === Configuration ===

# --- LLM Selection ---
# Read from environment variables. Ensure exactly ONE is set to 'True' or '1'.
# If none are explicitly set to True, Ollama will be the default.
USE_OPENAI = os.getenv("USE_OPENAI", "False").lower() in ('true', '1')
USE_OLLAMA = os.getenv("USE_OLLAMA", "False").lower() in ('true', '1')
USE_GEMINI = os.getenv("USE_GEMINI", "False").lower() in ('true', '1')

# Enforce single LLM selection
llm_selection_count = sum([USE_OPENAI, USE_OLLAMA, USE_GEMINI])
if llm_selection_count == 0:
    print("❗ No LLM explicitly selected in .env file, defaulting to Ollama.")
    USE_OLLAMA = True
elif llm_selection_count > 1:
    print("❌ ERROR: Multiple LLMs selected (USE_OPENAI, USE_OLLAMA, USE_GEMINI).")
    print("   Please set exactly ONE of these to 'True' in your .env file.")
    sys.exit(1)

# --- API Keys & Model Names (from .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4") # Default if not set

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3") # Default if not set
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-pro") # Default if not set

# --- Debate Settings ---
NUM_REBUTTAL_ROUNDS = int(os.getenv("NUM_REBUTTAL_ROUNDS", 2))
DEBATE_LOG_FILE = os.getenv("DEBATE_LOG_FILE", "debate_log.txt")
SLEEP_TIME_BETWEEN_TURNS = int(os.getenv("SLEEP_TIME_BETWEEN_TURNS", 2)) # Slightly longer default

# --- Initialize Faker ---
fake = Faker()

# === Helper Functions ===

def check_ollama_alive(url: str) -> bool:
    """Checks if the Ollama server is running and accessible."""
    try:
        response = requests.get(f"{url}/api/tags") # More reliable endpoint
        response.raise_for_status()
        print(f"✅ Ollama server connection successful at {url}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Ollama server not found at {url}. Is it running?")
        print("   Ensure Ollama is installed and running (`ollama serve`).")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to Ollama server at {url}: {e}")
        return False

def get_llm():
    """Initializes and returns the selected LangChain LLM instance."""
    if USE_OPENAI:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
            print("❌ ERROR: OpenAI selected, but OPENAI_API_KEY is not set in .env file.")
            sys.exit(1)
        print(f"⚙️ Using OpenAI model: {OPENAI_MODEL_NAME}")
        try:
            # Using ChatOpenAI as it's often preferred over the base OpenAI LLM class
            return ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.7, api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize OpenAI LLM: {e}")
            sys.exit(1)

    elif USE_OLLAMA:
        if not check_ollama_alive(OLLAMA_BASE_URL):
            sys.exit(1)
        print(f"⚙️ Using Ollama model: {OLLAMA_MODEL_NAME} from {OLLAMA_BASE_URL}")
        try:
            # Using the base Ollama class from langchain_community
            return Ollama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Ollama LLM: {e}")
            sys.exit(1)

    elif USE_GEMINI:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-google-api-key-here":
            print("❌ ERROR: Google Gemini selected, but GEMINI_API_KEY is not set in .env file.")
            sys.exit(1)
        print(f"⚙️ Using Google Gemini model: {GEMINI_MODEL_NAME}")
        try:
            # Ensure necessary packages are installed (listed in requirements.txt)
            return ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0.7,
                                          convert_system_message_to_human=True) # Often needed for compatibility
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Google Gemini LLM: {e}")
            print("   Ensure you have installed 'langchain-google-genai' and 'google-generativeai'.")
            sys.exit(1)

    else:
        # This case should theoretically not be reached due to earlier checks
        print("❌ INTERNAL ERROR: No valid LLM provider determined.")
        sys.exit(1)

def generate_debater_identity() -> dict:
    """Generates a fictional debater's identity using Faker."""
    name = fake.name()
    job = fake.job()
    # Generate more diverse and slightly longer personality traits
    personality_traits = [fake.catch_phrase() for _ in range(random.randint(1, 2))]
    personality = ". ".join(personality_traits).capitalize() + "."
    # Generate a more goal-oriented phrase
    goal = f"To champion the perspective of {fake.bs()}."
    backstory = (
        f"Hailing from {fake.city()}, {name} built a reputation as a {job}. "
        f"Known for being {fake.word()} yet {fake.word()}, they believe that {fake.sentence(nb_words=10)} "
        f"Their defining experience was {fake.paragraph(nb_sentences=1)}"
    )
    return {
        "name": name,
        "job": job,
        "personality": personality,
        "goal": goal,
        "backstory": backstory
    }

def run_debate_turn(agent: Agent, agent_identity: dict, opponent_statement: str | None, topic: str, turn_description: str) -> str:
    """Runs a single turn for an agent, handling potential LLM errors."""
    if opponent_statement:
        opponent_name = opponent_statement.split(':')[0] if ':' in opponent_statement else 'your opponent'
        opponent_text = opponent_statement.split(':', 1)[1].strip() if ':' in opponent_statement else opponent_statement
        prompt = (
            f"You are {agent_identity['name']}, a {agent_identity['job']}. Your personality: {agent_identity['personality']}. Your goal: {agent_identity['goal']}.\n"
            f"Debate Topic: '{topic}'\n\n"
            f"Your opponent, {opponent_name}, just presented this argument:\n\"\"\"\n{opponent_text}\n\"\"\"\n\n"
            f"Now, deliver your {turn_description}. Make a clear point, directly address their argument if possible, and stay in character."
        )
    else:
        # Opening statement
        prompt = (
             f"You are {agent_identity['name']}, a {agent_identity['job']}. Your personality: {agent_identity['personality']}. Your goal: {agent_identity['goal']}.\n"
             f"Debate Topic: '{topic}'\n\n"
             f"Deliver your powerful {turn_description}. State your main position clearly and persuasively, establishing your stance from the outset. Stay in character."
        )

    print(f"\n--- {agent_identity['name']} ({turn_description}) ---")
    statement = "Error: Could not generate statement." # Default error message
    try:
        # Use agent.invoke for potentially better compatibility/control in newer CrewAI versions
        # Fallback to agent.run if invoke isn't available or doesn't work as expected
        if hasattr(agent, 'invoke'):
            result = agent.invoke({"prompt": prompt}) # Check CrewAI docs for exact invoke input format
            statement = str(result) # Adjust based on actual return type of invoke
        else:
             # Older CrewAI might use agent.run
             result = agent.run(prompt)
             statement = str(result)

        print(f"{agent_identity['name']}:\n{statement}\n")
        time.sleep(SLEEP_TIME_BETWEEN_TURNS)

    except Exception as e:
        print(f"❌ ERROR during {agent_identity['name']}'s turn: {e}")
        print("   Skipping turn due to error.")
        # You might want to implement retry logic here

    # Ensure statement is a string before prepending name
    if not isinstance(statement, str):
        statement = str(statement) # Convert if necessary (e.g., if invoke returned an object)

    return f"{agent_identity['name']}: {statement}" # Prepend name for history


def log_debate(filename: str, topic: str, agent_1_id: dict, agent_2_id: dict, history: list, llm_used: str):
    """Logs the debate details and transcript to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = "\n\n".join(history) # Add blank line between turns

    log_content = f"""
=================================================
 Debate Log: {timestamp}
=================================================
LLM Used: {llm_used}
Topic: {topic}

--- Debater 1 ---
Name:        {agent_1_id['name']}
Role:        {agent_1_id['job']}
Goal:        {agent_1_id['goal']}
Personality: {agent_1_id['personality']}
Backstory:   {agent_1_id['backstory']}

--- Debater 2 ---
Name:        {agent_2_id['name']}
Role:        {agent_2_id['job']}
Goal:        {agent_2_id['goal']}
Personality: {agent_2_id['personality']}
Backstory:   {agent_2_id['backstory']}

-------------------------------------------------
 Debate Transcript
-------------------------------------------------
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
    # Determine which LLM was actually initialized for logging
    if USE_OPENAI: llm_info = f"OpenAI ({OPENAI_MODEL_NAME})"
    elif USE_OLLAMA: llm_info = f"Ollama ({OLLAMA_MODEL_NAME} @ {OLLAMA_BASE_URL})"
    elif USE_GEMINI: llm_info = f"Google Gemini ({GEMINI_MODEL_NAME})"
    else: llm_info = "Unknown" # Should not happen

    # --- Define Debate Topic ---
    user_topic = input(f"Enter a debate topic (or press Enter for a random one generated by Faker): ").strip()
    topic = user_topic if user_topic else f"Should {fake.company()} be regulated for its impact on {fake.bs()}?"
    print(f"\n💬 Debate Topic: {topic}\n")

    # --- Generate Debaters ---
    print("🎭 Generating Debater Identities...")
    identity_1 = generate_debater_identity()
    identity_2 = generate_debater_identity() # Ensure diversity

    print(f"👤 Debater 1: {identity_1['name']} ({identity_1['job']})")
    print(f"   Goal: {identity_1['goal']}")
    # print(f"   Backstory Snippet: {identity_1['backstory'][:100]}...") # Optional: show snippet
    print(f"👤 Debater 2: {identity_2['name']} ({identity_2['job']})")
    print(f"   Goal: {identity_2['goal']}")
    # print(f"   Backstory Snippet: {identity_2['backstory'][:100]}...") # Optional: show snippet

    # --- Create CrewAI Agents ---
    print("\n🤖 Creating CrewAI Agents...")
    try:
        agent_1 = Agent(
            role=identity_1['job'],
            goal=f"Win the debate about '{topic}'. Your ultimate objective is: {identity_1['goal']}",
            backstory=(
                f"You are {identity_1['name']}, a renowned {identity_1['job']}. "
                f"Your personality is characterized by: {identity_1['personality']}. "
                f"Your relevant background includes: {identity_1['backstory']}"
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False # Debaters shouldn't delegate
        )

        agent_2 = Agent(
            role=identity_2['job'],
            goal=f"Win the debate about '{topic}'. Your ultimate objective is: {identity_2['goal']}",
            backstory=(
                f"You are {identity_2['name']}, an experienced {identity_2['job']}. "
                f"Your personality is characterized by: {identity_2['personality']}. "
                f"Your relevant background includes: {identity_2['backstory']}"
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to create CrewAI agents: {e}")
        sys.exit(1)

    # --- Run the Debate ---
    debate_history = []
    last_statement = None # Start with no previous statement

    print("\n=== Debate Starting ===")

    # Opening Statements
    statement_1 = run_debate_turn(agent_1, identity_1, last_statement, topic, "Opening Statement")
    debate_history.append(statement_1)
    last_statement = statement_1

    statement_2 = run_debate_turn(agent_2, identity_2, last_statement, topic, "Opening Statement / Initial Rebuttal")
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

    print("\n=== Debate Concluded ===")

    # --- Log Debate ---
    log_debate(DEBATE_LOG_FILE, topic, identity_1, identity_2, debate_history, llm_info)

    print("\n✅ Debate finished and logged successfully.")
