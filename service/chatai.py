from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from service.knowledge_base import KnowledgeBase
from utils.config import Config

class TerraChatAI:
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=config.OPENAI_API_KEY
        )
        self.kb = KnowledgeBase(config)
        self.setup_rag_chain()

    def setup_rag_chain(self):
        template = """Ты ассистент TERRA. Отвечай на основе контекста:
        
        Контекст:
        {context}
        
        Вопрос:
        {question}
        
        Ответ (четко и по делу):"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {"context": self._retrieve_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def _retrieve_context(self, question: str) -> str:
        docs = await self.kb.search(question, k=3)
        return "\n\n".join(
            f"Вопрос: {doc.page_content}\nОтвет: {doc.metadata.get('answer', '')}"
            for doc in docs
        )

    async def generate_answer(self, question: str) -> str:
        return await self.rag_chain.ainvoke(question)