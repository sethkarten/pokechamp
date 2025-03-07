from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os

class GPTPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-4o', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None) -> str:
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()
        try:
            if json_format:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"},
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    # seed=seed,
                    stop=stop,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    stop=stop,
                    max_tokens=max_tokens
                )
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions)
        outputs = response.choices[0].message.content
        # log completion tokens
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        if json_format:
            return outputs, True
        return outputs, False
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()
        try:
            output_padding = ''
            if json_format:
                output_padding  = '\n{"'
                
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt+output_padding}
                ],
                temperature=temperature,
                stream=False,
                stop=stop,
                max_tokens=max_tokens
            )
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        
        if json_format:
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        return message, False
