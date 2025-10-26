from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os
import json

class OpenRouterPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENROUTER_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0
        
        # Optional headers for OpenRouter leaderboards
        self.site_url = os.getenv('OPENROUTER_SITE_URL', 'https://github.com/pokechamp')
        self.site_name = os.getenv('OPENROUTER_SITE_NAME', 'PokeChamp')

    def get_LLM_action(self, system_prompt, user_prompt, model='openai/gpt-4o', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, battle=None, ps_client=None) -> str:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        try:
            if json_format:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"},
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    stop=stop,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
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
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                )
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client)
        except Exception as e:
            print(f'OpenRouter API error: {e}')
            # sleep 2 seconds and try again
            sleep(2)
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client)
            
        outputs = response.choices[0].message.content
        
        # log completion tokens
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        
        if json_format:
            # Handle cases where the model adds extra text before the JSON
            # Look for the first { and last } to extract JSON
            start_idx = outputs.find('{')
            end_idx = outputs.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = outputs[start_idx:end_idx + 1]
                try:
                    # Validate JSON
                    json.loads(json_content)
                    return json_content, True, outputs  # Return processed, json_flag, raw
                except json.JSONDecodeError:
                    # If JSON is invalid, return the original output
                    return outputs, True, outputs  # Return processed, json_flag, raw
            else:
                # No JSON found, return original output
                return outputs, True, outputs  # Return processed, json_flag, raw
        
        return outputs, False, outputs  # Return processed, json_flag, raw
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='openai/gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
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
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            )
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        except Exception as e:
            print(f'OpenRouter API error: {e}')
            # sleep 2 seconds and try again
            sleep(2)
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        
        if json_format:
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        return message, False 