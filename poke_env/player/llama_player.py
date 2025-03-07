# import ollama
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
    
class LLAMAPlayer():
    def __init__(self, model="meta-llama/Meta-Llama-3-8B-Instruct", device=3) -> None:
        # self.model = model
        # self.client = ollama.Client(host="http://0.0.0.0:11434/api/chat")
        
        # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_id = "meta-llama/Llama-3.2-1B"
        # model_id = "/data2/milkkarten/pokemon-LLM/train/checkpoints/llama1b_SFT_oct28_elo/checkpoint-5500"
        # model_id = "/data2/milkkarten/pokemon-LLM/train/checkpoints/llama1b_SFT_oct28_elo_mix1/checkpoint-20000"
        # model_id = "/data2/milkkarten/pokemon-LLM/train/checkpoints/llama1b_SFT_oct28_elo_corrected/checkpoint-45000"
        # model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        # model_id "= model
        model_id = "meta-llama/Llama-3.2-3B"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            # quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        )
        self.model = torch.compile(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=True, seed=None, stop=[], max_tokens=20, actions=None) -> str:
        # assert json_format
        # # send message
        # response = self.client.generate(
        #     model='llama3:8b', 
        #     prompt=user_prompt,
        #     system=system_prompt,
        #     format='json',
        #     stream=False,
        #     options = {
        #             "temperature": temperature,
        #             "stop": stop,
        #             "num_predict": max_tokens,
        #         }
        # )
        # message = response['response']
        output_padding = ''
        if json_format:
            output_padding  = '\n{"'
        inputs = self.tokenizer(system_prompt+user_prompt+output_padding, return_tensors='pt').to(f'cuda:{self.device}')
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, pad_token_id=self.tokenizer.eos_token_id)
        response = generated_ids[0][inputs['input_ids'].shape[-1]:]
        message = self.tokenizer.decode(response, skip_special_tokens=True)
        if json_format:
            # json_start = message.find('{"')
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                print('output:', message_json)
                return message_json, True
        return message, False
    
    def get_LLM_action_topK_single(
        self, system_prompt, user_prompt, model, actions, top_k=3, temperature=0.7, 
        json_format=True, seed=None, stop=[], max_tokens=50, id_included=False
    ) -> list:

        # Prepare inputs
        output_padding = ''
        if json_format:
            output_padding = '\n{"'
        inputs = self.tokenizer(system_prompt + user_prompt + output_padding, return_tensors='pt').to(f'cuda:{self.device}')

        # Generate logits and output
        generated_outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        logits = torch.stack(generated_outputs.scores, dim=1).to('cpu')  # Shape: [seq_len, vocab_size]
        logits = logits[0]
        generated_ids = generated_outputs.sequences[0]
        
        # Slice out only the response portion
        response_start = inputs['input_ids'].shape[-1]
        response_ids = generated_ids[response_start:]
        message = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        # print("output message:", message)
        # Calculate probabilities for each action - player
        action_probabilities = {}
        for action in actions:
            if not id_included:
                action = action[2:]
            # print(action)
            # Tokenize the action string
            action_ids = self.tokenizer(action, return_tensors='pt')['input_ids'].squeeze(0)
            
            # Ensure the action length does not exceed generated logits
            # print(action_ids.shape, logits.shape)
            if action_ids.shape[0] > logits.shape[0]:
                action_probabilities[action] = 0.0
                continue

            # Compute the probability for the entire action string
            prob = 0
            for t, token_id in enumerate(action_ids[1:]):
                # print(token_id, response_ids[t])
                token_logits = logits[t]  # Logits for timestep t
                token_probs = F.softmax(token_logits, dim=-1)  # Convert logits to probabilities
                # print(token_probs[token_id])
                prob += token_probs[token_id].item()  # Multiply probabilities for each token

            action_probabilities[action] = prob

        # Sort actions by probability in descending order
        sorted_actions = sorted(action_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Return the top K actions
        print("sorted", sorted_actions[:top_k])
        output = ['{"' + action[0] for action in sorted_actions[:top_k]]
        print("sorted", output)
        return output
    
    def get_LLM_action_topK(
        self, system_prompt, user_prompt, model, actions, actions_opp, top_k=3, temperature=0.7, 
        json_format=True, seed=None, stop=[], max_tokens=50
    ) -> list:

        # Prepare inputs
        output_padding = ''
        if json_format:
            output_padding = '\n{"'
        inputs = self.tokenizer(system_prompt + user_prompt + output_padding, return_tensors='pt').to(f'cuda:{self.device}')

        # Generate logits and output
        generated_outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        logits = torch.stack(generated_outputs.scores, dim=1).to('cpu')  # Shape: [seq_len, vocab_size]
        logits = logits[0]
        generated_ids = generated_outputs.sequences[0]
        
        # Slice out only the response portion
        response_start = inputs['input_ids'].shape[-1]
        response_ids = generated_ids[response_start:]
        message = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        action_player_tokens = message[:message.index(',')+2]
        player_tokens = self.tokenizer(action_player_tokens, return_tensors='pt')['input_ids'].squeeze(0)
        action_player_index = len(player_tokens)
        print("output message:", message)
        print(action_player_index, message[:message.index(',')+2])
        # Calculate probabilities for each action - player
        action_probabilities = {}
        for action in actions:
            action = action[2:]
            # print(action)
            # Tokenize the action string
            action_ids = self.tokenizer(action, return_tensors='pt')['input_ids'].squeeze(0)
            
            # Ensure the action length does not exceed generated logits
            # print(action_ids.shape, logits.shape)
            if action_ids.shape[0] > logits.shape[0]:
                action_probabilities[action] = 0.0
                continue

            # Compute the probability for the entire action string
            prob = 0
            for t, token_id in enumerate(action_ids[1:]):
                # print(token_id, response_ids[t])
                token_logits = logits[t]  # Logits for timestep t
                token_probs = F.softmax(token_logits, dim=-1)  # Convert logits to probabilities
                # print(token_probs[token_id])
                prob += token_probs[token_id].item()  # Multiply probabilities for each token

            action_probabilities[action] = prob
            
        if len(actions_opp) != 0:
            # Calculate probabilities for each action - player
            action_probabilities_opp = {}
            for action in actions_opp:
                action = action[2:]
                # print(action)
                # Tokenize the action string
                action_ids = self.tokenizer(action, return_tensors='pt')['input_ids'].squeeze(0)
                # Ensure the action length does not exceed generated logits
                # print(action_ids.shape, logits.shape)
                if action_ids.shape[0] > logits.shape[0]:
                    action_probabilities_opp[action] = 0.0
                    continue

                # Compute the probability for the entire action string
                prob = 0
                for t, token_id in enumerate(action_ids[2:]):
                    # print(token_id, response_ids[t])
                    token_logits = logits[t+action_player_index]  # Logits for timestep t
                    token_probs = F.softmax(token_logits, dim=-1)  # Convert logits to probabilities
                    # print(token_probs[token_id])
                    prob += token_probs[token_id].item()  # Multiply probabilities for each token

                action_probabilities_opp[action] = prob

        # Sort actions by probability in descending order
        sorted_actions = sorted(action_probabilities.items(), key=lambda x: x[1], reverse=True)
        sorted_actions_opp = sorted(action_probabilities_opp.items(), key=lambda x: x[1], reverse=True)

        # Return the top K actions
        print("sorted", sorted_actions[:top_k], sorted_actions_opp[:top_k])
        print(actions_opp)
        output = ['{"' + action[0] for action in sorted_actions[:top_k]]
        output_opp = ['{"' + action[0] for action in sorted_actions_opp[:top_k]]
        # print("sorted", output)
        return output, output_opp


    def get_LLM_action_state_value(self, system_prompt, user_prompt, model, temperature=0.7, json_format=True, seed=None, stop=[], max_tokens=20, actions=None) -> str:
        output_padding = ''
        if json_format:
            output_padding = '\n{"'
        inputs = self.tokenizer(system_prompt + user_prompt + output_padding, return_tensors='pt').to(f'cuda:{self.device}')

        # Generate tokens with scores enabled
        generated_outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, 
                                                pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)

        # Extract token IDs and corresponding logits
        generated_ids = generated_outputs.sequences[0]
        logits = torch.stack(generated_outputs.scores, dim=1).to('cpu')  # Stack along sequence to get scores for each token

        # Slice out only the response portion
        response_ids = generated_ids[inputs['input_ids'].shape[-1]:]
        # Decode the response to find the "<winner>" token positions
        message = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        print(message)
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities

        # Locate the start and end of "<winner>" tokens in response
        response_ids = response_ids.to('cpu')
        # token_start = self.tokenizer.tokenize('":"')
        token_end = self.tokenizer.tokenize('"}')
        token_start = 3332
        token_end = 9388
        winner_token = 53245    # winner
        end = (response_ids == token_end).nonzero(as_tuple=True)[0][-1].item()
        start = (response_ids[:end] == token_start).nonzero(as_tuple=True)[0][-1].item() + 1
        winner_token_ids = response_ids[start:end].tolist()
        print(winner_token_ids, start, end)
        # Extract probabilities for "<winner>" tokens
        winner_prob = 1
        winner = 'opponent'
        for t, token_id in zip(np.arange(start, end), winner_token_ids):
            if token_id == 3517:
                winner = 'player'
            if token_id == 3517 or token_id == 454 or token_id == 1166:
                print('compare', token_id, probs[0, t, 3517].item(), probs[0, t, 454].item(), probs[0, t, 1166].item())
            token_prob = probs[0, t, token_id].item()
            winner_prob *= token_prob
            # winner_probs.append((self.tokenizer.decode([token_id]), token_prob))
            # print(token_id, winner_prob)
        if winner == 'opponent':
            winner_prob = 1 - winner_prob

        # Print probabilities for debugging or further use
        winner_probs = []
        # winner_token_ids = response_ids[0:].tolist()
        # print(winner_token_ids)
        for t, token_id in zip(np.arange(start, end), winner_token_ids):
            token_prob = probs[0, t, token_id].item()
            winner_probs.append((token_id, token_prob))
        for token, prob in winner_probs:
            print(f"Token: '{token}' - Probability: {prob}")
        # JSON handling
        if json_format:
            json_start = 0
            json_end = message.find('}') + 1
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True, winner_prob
        return message, False, winner_prob
    
    def get_LLM_action_state_value_top_K(self, system_prompt, user_prompt, model, temperature=0.7, json_format=True, seed=None, stop=[], max_tokens=20, actions=None, top_K=3, dmg_calc_action=None):
        # first get top K actions
        actions_top_k = self.get_LLM_action_topK_single(system_prompt, user_prompt, model, actions, top_k=top_K, temperature=temperature, 
                                                        json_format=json_format, seed=seed, stop=stop, max_tokens=max_tokens, id_included=True)
        print(actions_top_k)
        if dmg_calc_action is not None:
            if dmg_calc_action not in actions_top_k:
                actions_top_k[-1] = dmg_calc_action
        outputs = []
        for action_player in actions_top_k:
            output_padding = ''
            if json_format:
                output_padding = '\n' + action_player
            inputs = self.tokenizer(system_prompt + user_prompt + output_padding, return_tensors='pt').to(f'cuda:{self.device}')

            # Generate tokens with scores enabled
            generated_outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, 
                                                    pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)

            # Extract token IDs and corresponding logits
            generated_ids = generated_outputs.sequences[0]
            logits = torch.stack(generated_outputs.scores, dim=1).to('cpu')  # Stack along sequence to get scores for each token

            # Slice out only the response portion
            response_ids = generated_ids[inputs['input_ids'].shape[-1]:]
            # Decode the response to find the "<winner>" token positions
            message = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            # print(message)
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities

            # Locate the start and end of "<winner>" tokens in response
            response_ids = response_ids.to('cpu')
            # token_start = self.tokenizer.tokenize('":"')
            token_end = self.tokenizer.tokenize('"}')
            token_start = 3332
            token_end = 9388
            winner_token = 53245    # winner
            end = (response_ids == token_end).nonzero(as_tuple=True)[0][-1].item()
            start = (response_ids[:end] == token_start).nonzero(as_tuple=True)[0][-1].item() + 1
            winner_token_ids = response_ids[start:end].tolist()
            # print(winner_token_ids, start, end)
            # Extract probabilities for "<winner>" tokens
            winner_prob = 1
            winner = 'opponent'
            for t, token_id in zip(np.arange(start, end), winner_token_ids):
                if token_id == 3517:
                    winner = 'player'
                if token_id == 3517 or token_id == 454 or token_id == 1166:
                    print('compare', token_id, probs[0, t, 3517].item(), probs[0, t, 454].item(), probs[0, t, 1166].item())
                token_prob = probs[0, t, token_id].item()
                winner_prob *= token_prob
            if winner == 'opponent':
                winner_prob = 1 - winner_prob

            # JSON handling
            if json_format:
                json_start = 0
                json_end = message.find('}') + 1
                message_json = action_player + message[json_start:json_end]
                # print('json', message_json)
                if len(message_json) > 0:
                    outputs.append((message_json, winner_prob))
        return outputs
