from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Generate two random agent identities
agent_a = {
    "name": fake.name(),
    "occupation": fake.job(),
    "philosophy": fake.catch_phrase()
}

agent_b = {
    "name": fake.name(),
    "occupation": fake.job(),
    "philosophy": fake.catch_phrase()
}

# Generate a surreal debate prompt using Faker
debate_prompt = f"What are the philosophical consequences when {fake.catch_phrase().lower()} intersects with {fake.bs().lower()} during {fake.job().lower()}?"

# Simulate their opening arguments
agent_a_argument = (
    f"{agent_a['name']}, a {agent_a['occupation'].lower()}, opens with the belief that '{agent_a['philosophy']}', "
    f"and argues that this principle fundamentally shapes the way we must interpret the question: \"{debate_prompt}\""
)

agent_b_argument = (
    f"{agent_b['name']}, a {agent_b['occupation'].lower()}, counters by stating that '{agent_b['philosophy']}', "
    f"which reframes the entire philosophical landscape surrounding the prompt: \"{debate_prompt}\""
)

agent_a, agent_b, debate_prompt, agent_a_argument, agent_b_argument
