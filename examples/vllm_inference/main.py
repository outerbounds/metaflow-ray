from metaflow import Runner

chat = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {
        "role": "user",
        "content": "Who are you?"
    },
]

with Runner(flow_file="./flow.py", pylint=False, environment="fast-bakery") as runner:
    result = runner.run(messages=chat)
