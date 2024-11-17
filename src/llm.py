import os
import base64
import requests
from json import JSONDecodeError
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def check_env_file():
    if not os.path.exists(".env"):
        raise FileNotFoundError("Please create a .env file with OPENAI_API")
    with open(".env", "r") as f:
        env_vars = f.readlines()
    for env_var in env_vars:
        if "OPENAI_API" in env_var:
            break
    else:
        raise ValueError("Please add OPENAI_API to the .env file.")
    return

def generate_captions(
    llm: str,
    clsname: str,
    num_real_samples: int,
    is_visual: bool = True,
    image_paths: list = None,
):
    check_env_file()
    # load .env file
    load_dotenv()
    
    client = OpenAI()
    if not is_visual:
        while True:
            response = client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": f"Generate captions for {clsname} images. Specify detail visual attributes. You have to generate {num_real_samples} captions seperated by '|'. For example, 'A cat with blue eyes and white fur.'"},
                ]
            )
            output = response.choices[0].message.content
            gen_captions = output.split("|")
            if len(gen_captions) == num_real_samples:
                break
            else:
                print("Re-try generating captions.")
    else:
        assert image_paths is not None, "Please provide image paths."
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        }
        gen_captions = []
        for path in tqdm(image_paths):
            while True:
                base64_image = encode_image(path)
                payload = {
                    "model": f"{llm}",
                    "messages": [
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": "Please generate a caption for this image. Specify detail visual attributes. Describe the image in detail."
                            },
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                            }
                        ]
                        }
                    ],
                    "max_tokens": 300
                }
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
                )
                try:
                    output = response.json()["choices"][0]["message"]["content"]
                    break
                except JSONDecodeError:
                    print(response.json())
                    print("Re-try generating captions.")
                    continue
            output = response.json()["choices"][0]["message"]["content"]
            gen_captions.append(output)
    return gen_captions

if __name__ == "__main__":
    llm = "gpt-4o-mini"
    clsname = "cat"
    num_real_samples = 5
    is_visual = False
    gen_captions = generate_captions(llm, clsname, num_real_samples, is_visual)# , image_paths)
    print(gen_captions)