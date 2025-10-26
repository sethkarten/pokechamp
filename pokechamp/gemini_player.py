from google import genai
from time import sleep
import os, sys
import json

class GeminiPlayer():
    def __init__(self, api_key=""):
        print("api_key", api_key)
        if api_key == "":
            self.api_key = os.getenv('GEMINI_API_KEY')
        else:
            self.api_key = api_key
        
        # Configure the Gemini API
        self.client = genai.Client(api_key=self.api_key)
        
        self.completion_tokens = 0
        self.prompt_tokens = 0
        
        # Map common model names to official API names
        self.model_mapping = {
            # Default to latest Gemini 2.5
            'gemini-flash': 'gemini-2.5-flash',
            'gemini-flash-2.5': 'gemini-2.5-flash',
            'gemini-pro': 'gemini-2.5-pro',
            'gemini-pro-2.5': 'gemini-2.5-pro',
            
            # Gemini 2.0 models
            'gemini-2.0-flash': 'gemini-2.0-flash',
            'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite',
            'gemini-2.0-pro': 'gemini-2.0-pro-experimental',
            'gemini-2.0-pro-experimental': 'gemini-2.0-pro-experimental',
            'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking-exp',
            
            # Gemini 1.5 models (legacy support)
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-1.5-pro': 'gemini-1.5-pro',
        }

    def get_LLM_action(self, system_prompt, user_prompt, model='gemini-2.0-flash', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=1000, actions=None, battle=None, ps_client=None) -> str:
        try:
            # Map model name to official API name
            api_model_name = self.model_mapping.get(model, model)
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # print("=" * 80)
            # print("GEMINI PROMPT:")
            # print("-" * 80)
            # print(combined_prompt)
            # print("-" * 80)
            
            # Generate response
            response = self.client.models.generate_content(model=api_model_name, contents=combined_prompt)
            
            # print("GEMINI RESPONSE:")
            # print("-" * 80)
            # print(response)
            # print("-" * 80)
            
            # Extract text from response
            outputs = response.text
            
            # Simple token counting approximation (Gemini doesn't provide exact counts)
            self.completion_tokens += len(outputs.split()) * 1.3  # Approximate tokens
            self.prompt_tokens += len(combined_prompt.split()) * 1.3
            
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
            
        except Exception as e:
            print(f'Gemini API error: {e}')
            sys.exit(1)
            # sleep 2 seconds and try again
            sleep(2)
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client)
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gemini-2.0-flash', json_format=False, seed=None, stop=[], max_tokens=1000):
        try:
            # Map model name to official API name
            api_model_name = self.model_mapping.get(model, model)
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate response
            response = self.client.models.generate_content(model=api_model_name, contents=combined_prompt)
            
            # Extract text from response
            message = response.text
            
        except Exception as e:
            print(f'Gemini API error2: {e}')
            sys.exit(1)
            # sleep 2 seconds and try again
            sleep(2)
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        
        if json_format:
            json_start = 0
            json_end = message.find('}') + 1  # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        
        return message, False