import ollama
import json
import numpy as np
import time

class OllamaPlayer():
    def __init__(self, model="llama3.1:8b", device=None) -> None:
        """
        Initialize the Ollama player using Ollama API.
        
        Args:
            model: The Ollama model name (e.g., "llama3.1:8b", "gpt-oss:20b", etc.)
            device: Not used with Ollama API, kept for compatibility
        """
        self.model = model
        
        # Configuration for Ollama client
        self.base_url = "http://localhost:11434"
        self.request_timeout = 300  # 5 minutes timeout
        self.temperature = 0.7
        self.context_window = 8192  # Use larger context window for pokechamp context
        self.max_tokens = 8192  # Limit response length but ensure complete JSON
        
        # Initialize client with configuration
        self.client = ollama.Client(host=self.base_url)
        
        # Check if model is available
        # try:
        #     models = self.client.list()
        #     model_names = [m['name'] for m in models.get('models', [])]
        #     if not any(self.model in name for name in model_names):
        #         print(f"Warning: Model {self.model} not found. Available models: {model_names}")
        #         print(f"You may need to run: ollama pull {self.model}")
        # except Exception as e:
        #     print(f"Warning: Could not check available models: {e}")
    
    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=True, seed=None, stop=[], max_tokens=20, actions=None, think=True, battle=None, ps_client=None) -> str:
        """
        Get action from LLM using Ollama API.
        
        Args:
            think: Whether to enable thinking mode for models that support it
        """
        # if 'qwen3' in self.model.lower() or 'oss' in self.model.lower():
        #     user_prompt = user_prompt + '\nDo not think, just answer.'
        
        # Prepare the prompt - add JSON formatting
        if json_format:
            user_prompt_with_json = user_prompt + '\n{"'
        else:
            user_prompt_with_json = user_prompt
        
        # Set up generation options
        options = {
            'temperature': temperature if temperature != 0.7 else self.temperature,
            'num_predict': max(max_tokens, self.max_tokens),
            'num_ctx': self.context_window,
        }
        if seed is not None:
            options['seed'] = seed
        if stop:
            options['stop'] = stop
        
        # Add thinking mode parameter for models that support it
        # if 'qwen3' in self.model.lower() or 'oss' in self.model.lower():
        #     # Try multiple approaches to disable thinking
        #     options['think'] = False
        #     options['enable_thinking'] = False
        #     options['thinking'] = False
        #     print(f"Disabling thinking for {self.model}")
        
        try:
            # Use chat endpoint
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt_with_json}
            ]
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=False
            )
            
            # Extract message content
            message = ""
            thinking = ""
            
            if hasattr(response, 'message'):
                if hasattr(response.message, 'content'):
                    message = response.message.content
                if hasattr(response.message, 'thinking') and think:
                    thinking = response.message.thinking
            elif isinstance(response, dict):
                message = response.get('message', {}).get('content', '')
                if think:
                    thinking = response.get('message', {}).get('thinking', '')
            
            # Debug message content and thinking
            if thinking:
                print(f"=== THINKING ===")
                print(thinking)
                print("=" * 40)
            
            print(f'Message content: "{message}"')
            
            if json_format:
                # Extract JSON from response
                json_start = message.find('{"')
                if json_start >= 0:
                    json_part = message[json_start:]
                    json_end = json_part.find('}')
                    if json_end > 0:
                        message_json = json_part[:json_end + 1]
                        print('output:', message_json)
                        # Combine thinking and message for raw output
                        combined_raw = f"THINKING: {thinking}\n\nRESPONSE: {message}" if thinking else message
                        return message_json, True, combined_raw
                elif message.startswith('"'):
                    # Complete the JSON that started with '{"'
                    message_json = '{"' + message
                    json_end = message_json.find('}')
                    if json_end > 0:
                        message_json = message_json[:json_end + 1]
                        print('output:', message_json)
                        # Combine thinking and message for raw output
                        combined_raw = f"THINKING: {thinking}\n\nRESPONSE: {message}" if thinking else message
                        return message_json, True, combined_raw
                else:
                    # Look for any JSON-like pattern
                    import re
                    json_match = re.search(r'\{[^}]*\}', message)
                    if json_match:
                        message_json = json_match.group(0)
                        print('output:', message_json)
                        # Combine thinking and message for raw output
                        combined_raw = f"THINKING: {thinking}\n\nRESPONSE: {message}" if thinking else message
                        return message_json, True, combined_raw
            
            # Combine thinking and message for raw output
            combined_raw = f"THINKING: {thinking}\n\nRESPONSE: {message}" if thinking else message
            return message, False, combined_raw
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "", False, ""
    