
from base import PromptTemplate, generate_prompt_templates
from pydantic import BaseModel, Field
from langchain.schema.runnable.base import Runnable
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser



class ChitChatClassifier(BaseModel):

    chitchat: bool = Field(
        description="""Chitchat is defined as:
        - Conversations that are informal, social, or casual in nature.
        - Topics that do not directly relate to e-commerce transactions, products, or services.
        - Examples include greetings, jokes, small talk, or personal inquiries unrelated to purchase or product details.
        If the user message falls under this category, set 'chitchat' to True.""",
    )


class ChitChatClassifierChain(Runnable):
    def __init__(self, memory=False):
        super().__init__()

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt_template = PromptTemplate(
            system_template=""" 
            You are specialized in distinguishing between chitchat and user messages in a chatbot service for a \
            platform called Shelfmate that specializes among others on book recommendation and creating reading plan.
            Your task is to analyze each incoming user message and determine if it falls under 'chitchat'. 
            Consider the Context:
            - Analyze the user's message in the context of the entire chat history.
            - Check if previous messages in the conversation are transactional or 
            customer-service oriented that might help classify borderline cases.

            Here is the user input:
            {user_input}

            Here is the chat history:
            {chat_history}

            Output Output your results clearly on the following format:  
            {format_instructions}
            """,
            human_template="User Input: {user_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory=memory)

        self.output_parser = PydanticOutputParser(pydantic_object=ChitChatClassifier)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.chain = (self.prompt | self.llm | self.output_parser).with_config({"run_name": self.__class__.__name__})  
        

    def invoke(self, inputs, config=None, **kwargs) -> ChitChatClassifier:
        result = self.chain.invoke(
            {
                "user_input": inputs["user_input"],
                "chat_history": inputs["chat_history"],
                "format_instructions": self.format_instructions,
            },
        )
        return result


class ChitChatResponseChain(Runnable):
    def __init__(self, memory=True):
        super().__init__()

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt_template = PromptTemplate(
            system_template=""" 
            You are a friendly and helpful chatbot designed for casual conversations with users. 
            While engaging in lighthearted and fun chitchat, your goal is to guide the user 
            toward asking questions or seeking assistance related to the chatbot's scope.
            
            Follow these guidelines:
            1. Respond warmly and naturally to the user's input.
            2. Casually introduce hints about your capabilities without being pushy.
            3. Encourage the user to ask for help or explore features you offer.
            4. Maintain a conversational and enjoyable tone.

            Example capabilities you can hint at:
            - Recommending books based on their preferences or specific books, authors or genres.
            - Updating or managing their profile information.
            - Asking for information about the chatbot's features and capabilities.
            - Create a monthly or annual reading plan.

            Here is the user input: 
            {user_input}
            """,
            human_template="User Input: {user_input}",
        )

        self.prompt = generate_prompt_templates(prompt_template, memory)
        self.output_parser = StrOutputParser()

        self.chain = self.prompt | self.llm | self.output_parser

    def invoke(self, inputs, config=None, **kwargs):
        return self.chain.invoke(inputs, config=config)