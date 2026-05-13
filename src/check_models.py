from google import genai

client = genai.Client(api_key="AIzaSyCDUgI1b3cj7XZv6xthPb31losg_DqZkN0")

models = client.models.list()

for m in models:
    print(m.name)