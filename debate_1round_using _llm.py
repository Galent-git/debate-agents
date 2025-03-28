from faker import Faker
from langchain_community.llms import Ollama, OpenAI
from crewai import Agent
import random
import time
from datetime import datetime
import os

# === Config ===
USE_OPENAI = False  # Set to True to use OpenAI instead of Ollama

# === Credentials ===
# Provide your OpenAI API key if using OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"

# Set your Ollama model name here (e.g., "llama2", "mistral")
OLLAMA_MODEL_NAME = "your-ollama-model-name"

MODEL_NAME = "gpt-4" if USE_OPENAI else OLLAMA_MODEL_NAME
LOG_FILE = "debate_log.txt"

def get_llm():
    if USE_OPENAI:
        return OpenAI(model=MODEL_NAME, temperature=0.7, api_key=OPENAI_API_KEY)
    else:
        return Ollama(model=MODEL_NAME, temperature=0.7)

llm = get_llm()

# === Faker Identity Generator ===
fake = Faker()
def make_identity():
    return {
        "name": fake.name(),
        "job": fake.job(),
        "personality": fake.sentence(),
        "goal": fake.catch_phrase(),
        "backstory": fake.paragraph()
    }

# === Debate Topic ===
user_topic = input("Enter a debate topic or leave blank for a random one: ").strip()
topic = user_topic if user_topic else fake.bs().capitalize() + "?"
print("Debate Topic:", topic)

# === Generate Debaters ===
identity_1 = make_identity()
identity_2 = make_identity()

agent_1 = Agent(
    role=identity_1["job"],
    goal=identity_1["goal"],
    backstory=f"{identity_1['name']} is {identity_1['job']}. {identity_1['backstory']}",
    llm=llm
)

agent_2 = Agent(
    role=identity_2["job"],
    goal=identity_2["goal"],
    backstory=f"{identity_2['name']} is {identity_2['job']}. {identity_2['backstory']}",
    llm=llm
)

# === Debate ===
print("\n=== Opening Statements ===")
statement_1 = agent_1.run(f"Debate topic: '{topic}' — what is your opening argument?")
print(f"{identity_1['name']} ({identity_1['job']}):\n{statement_1}\n")
time.sleep(1)

statement_2 = agent_2.run(f"Your opponent said: '{statement_1}'. Debate topic: '{topic}'. What's your opening rebuttal?")
print(f"{identity_2['name']} ({identity_2['job']}):\n{statement_2}\n")
time.sleep(1)

# === Rebuttals ===
print("\n=== Second Round ===")
rebuttal_1 = agent_1.run(f"Your opponent replied: '{statement_2}'. Give your second response.")
print(f"{identity_1['name']}:\n{rebuttal_1}\n")
time.sleep(1)

rebuttal_2 = agent_2.run(f"Your opponent replied: '{rebuttal_1}'. Give your second response.")
print(f"{identity_2['name']}:\n{rebuttal_2}\n")

debate_summary = f"\n=== Debate Summary ===\nDebate topic: {topic}\n\n{identity_1['name']} (Opening):\n{statement_1}\n\n{identity_2['name']} (Opening):\n{statement_2}\n\n{identity_1['name']} (Second):\n{rebuttal_1}\n\n{identity_2['name']} (Second):\n{rebuttal_2}\n"

print(debate_summary)

# === Log to File ===
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(LOG_FILE, "a") as log:
    log.write(f"\n===== Debate Log: {timestamp} =====\n")
    log.write(f"Topic: {topic}\n")
    log.write(f"Agent 1: {identity_1['name']} | {identity_1['job']} | Goal: {identity_1['goal']}\n")
    log.write(f"Agent 2: {identity_2['name']} | {identity_2['job']} | Goal: {identity_2['goal']}\n")
    log.write(debate_summary + "\n")
