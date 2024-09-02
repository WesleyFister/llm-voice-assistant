from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def chatWithHistory(llm_model, transcription, chatHistory, systemPrompt):
    llm = Ollama(model=llm_model)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{systemPrompt}",
            ),
            MessagesPlaceholder(variable_name="chatHistory"),
            (
                "user",
                "{input}"
            ),
        ]
    )

    chain = prompt_template | llm
    
    response = chain.invoke({"input": transcription, "chatHistory": chatHistory, "systemPrompt": systemPrompt})
    chatHistory.append(HumanMessage(content=transcription))
    chatHistory.append(AIMessage(content=response))

    return chatHistory, response