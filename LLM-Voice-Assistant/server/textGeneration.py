from datetime import datetime
import json
import ollama
import nltk
from nltk.tokenize import sent_tokenize

def download_llm_model(llm_model):
    try:
        ollama.show(llm_model)
        
    except ollama._types.ResponseError:
        print(f'Downloading {llm_model} LLM model')
        ollama.pull(llm_model)
    
    nltk.download('punkt_tab')

def chatWithHistory(llm_model, input, chatHistoryFile, systemPrompt):
    systemPrompt = f'{systemPrompt} Current date: {datetime.now().strftime("%Y-%m-%d (%A)")} Current time: {datetime.now().strftime("%I:%M %p")}'
    
    try:
        with open(chatHistoryFile, "r") as file:
            json_string = file.read()

        # Convert the JSON string back to a Python list
        chatHistory = json.loads(json_string)

    except FileNotFoundError:
        # Convert the list to a JSON string
        json_string = json.dumps(
            [
                {
                'role': 'system',
                'content': systemPrompt,
                }
            ]
        )

        # Write the JSON string to the file
        with open(chatHistoryFile, 'w') as file:
            file.write(json_string)
            
        with open(chatHistoryFile, "r") as file:
            json_string = file.read()

        # Convert the JSON string back to a Python list
        chatHistory = json.loads(json_string)

    chatHistory.append(
        {
        'role': 'user',
        'content': input,
        }
    )
    stream = ollama.chat(model=llm_model, 
        messages=chatHistory,
        stream=True,
    )

    sentence = 1
    response = ""
    for chunk in stream:
        part = chunk['message']['content']
        #print(part, end='', flush=True)
        response = response + part
        
        sentences = sent_tokenize(response)

        if len(sentences) > sentence:
            yield sentences[sentence - 1]
            sentence += 1
    
    yield sentences[-1]

    chatHistory.append(
        {
        'role': 'assistant',
        'content': response,
        }
    )

    # Convert the list to a JSON string
    json_string = json.dumps(chatHistory)

    # Write the JSON string to the file
    with open(chatHistoryFile, 'w') as file:
        file.write(json_string)
