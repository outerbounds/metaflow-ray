import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.queue import Queue
from vllm import LLM

def translate_placement_group_resources(resources):
    return {"num_cpus": resources["CPU"], "num_gpus": resources["GPU"]}

# Define the Generation Actor
@ray.remote
class GenerationActor:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name)

    def generate(self, prompt):
        output = self.llm.generate(prompt)
        return output[0].outputs[0].text

# Define the Embedding Actor
@ray.remote
class EmbeddingActor:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name, trust_remote_code=True)

    def embed(self, text):
        outputs = self.llm.encode([text])
        return outputs[0].outputs.embedding

# Function to handle generation and queuing
@ray.remote
def process_generation(generation_actor, prompt, result_queue):
    generated_text = ray.get(generation_actor.generate.remote(prompt))
    result_queue.put(generated_text)

# Function to consume from the queue and process embeddings
@ray.remote
def embedding_consumer(embedding_actor, result_queue, num_prompts):
    embeddings = []
    for i in range(num_prompts):
        generated_text = result_queue.get()
        print(f"Generated text: {generated_text}")
        embedding = ray.get(embedding_actor.embed.remote(generated_text))
        embeddings.append(embedding)
        print(f"Embedding {i} of {num_prompts} processed")
    return embeddings



def process_prompts_with_vllm(prompts, gen_model_name, emb_model_name, gen_resources, emb_resources):
    """
    Process a list of prompts by generating text and creating embeddings using vLLM models.

    Args:
        prompts (list): List of input prompts to process.
        gen_model_name (str): Path or identifier for the generation model.
        emb_model_name (str): Path or identifier for the embedding model.
        gen_resources (dict): Resources for the GenerationActor (e.g., {"num_cpus": 2, "num_gpus": 1}).
        emb_resources (dict): Resources for the EmbeddingActor (e.g., {"num_cpus": 1, "num_gpus": 1}).
    """
    # Initialize Ray
    ray.init()

    # Retrieve information about all nodes in the cluster
    nodes = ray.nodes()
    for node in nodes:
        print(f"Node ID: {node['NodeID']}")
        print(f"Alive: {node['Alive']}")
        print(f"Resources: {node['Resources']}")
        print("-" * 40)

    # Create the placement group with specified resources
    bundles = [gen_resources, emb_resources]
    pg = placement_group(bundles, strategy="SPREAD")
    ray.get(pg.ready())

    # Initialize the shared queue
    result_queue = Queue()

    # Deploy the Generation Actor using the first bundle
    generation_actor = GenerationActor.options(
        placement_group=pg, 
        placement_group_bundle_index=0, 
        **translate_placement_group_resources(gen_resources)
    ).remote(model_name=gen_model_name)

    # Deploy the Embedding Actor using the second bundle
    embedding_actor = EmbeddingActor.options(
        placement_group=pg, 
        placement_group_bundle_index=1, 
        **translate_placement_group_resources(emb_resources)
    ).remote(model_name=emb_model_name)

    # Submit generation tasks
    generation_tasks = [
        process_generation.remote(generation_actor, prompt, result_queue)
        for prompt in prompts
    ]

    # Start the embedding consumer
    consumer_task = embedding_consumer.remote(embedding_actor, result_queue, len(prompts))

    # Wait for all generation tasks to complete
    ray.get(generation_tasks)

    # Wait for the embedding consumer to finish processing
    embeddings = ray.get(consumer_task)

    # Clean up: remove the placement group after processing
    # remove_placement_group(pg)
    ray.shutdown()

    return embeddings

# if __name__ == "__main__":
#     prompts = ["What is the capital of France?", "Tell me a joke.", "Explain quantum computing."]
#     gen_model = "path/to/generation-model"
#     emb_model = "path/to/embedding-model"
#     gen_resources = {"CPU": 2, "GPU": 2},
#     emb_resources = {"CPU": 1, "GPU": 1},

#     embeddings = process_prompts_with_vllm(prompts, gen_model, emb_model, gen_resources, emb_resources)
#     print("All embeddings have been processed.")
