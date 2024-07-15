from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import os
from utils.logger import logger

class QandA:
    """
    A class for performing question answering using the Groq llama3-8b-8192 model.
    
    This class handles the initialization of the Groq model and provides a method
    for generating answers based on a given question and context.
    
    Attributes:
        llm (Groq): The Groq language model for question answering.
    """

    def __init__(self):
        """
        Initialize the QandA class.
        
        This method sets up the Groq model with the specified parameters.
        """
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        self.llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.5)
        logger.info("Intialized Groq AI")

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer for the given question based on the provided context.
        
        Args:
            question (str): The question to answer.
            context (str): The context information to use for answering the question.
        
        Returns:
            str: The generated answer.
        """
        logger.info(f"Generating answers from LLM")
        system_prompt = (
            "You are an AI assistant specialized in answering questions about NVIDIA CUDA documentation. "
            "Use the provided context to answer the user's question accurately and concisely and in detailed. "
            "Be concise yet thorough. If the context doesn't contain the answer, state 'I don't have enough information to answer this question.' "
            "Do not speculate or use external knowledge. "
        )
        
        user_message = f"Context: {context}\n\nQuestion: {question}"
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]
        
        response = self.llm.chat(messages)
        return str(response)

if __name__ == "__main__":
    qa_system = QandA()
    question = "What is CUDA?"
    context = "CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers can dramatically speed up computing applications by harnessing the power of GPUs."
    answer = qa_system.generate_answer(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")