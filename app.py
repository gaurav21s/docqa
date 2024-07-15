import streamlit as st
from utils.web_crawl import run_spider
from utils.data import DataHandler
from utils.retrieve import Retrieval
from utils.qna import QandA
from utils.logger import logger

class DocQASystem:
    def __init__(self):
        self.crawler = run_spider
        self.data_handler = DataHandler()
        self.retriever = Retrieval()
        self.qa_system = QandA()

    def crawl_and_process(self):
        with st.spinner("Crawling NVIDIA CUDA documentation..."):
            self.crawler()
        
        with st.spinner("Processing and storing data..."):
            self.data_handler.process_and_store('output.json')

    def answer_question(self, question: str):
        # results_placeholder = st.empty()
        with st.spinner("Retrieving relevant documents..."):
            results = self.retriever.retrieve_and_rerank(question)
            # results_placeholder.write("Retrieved relevant documents.")

        # answer_placeholder = st.empty()
        with st.spinner("Generating answer..."):
            context = self.get_context(results)
            answer = self.qa_system.generate_answer(question, context)
            # answer_placeholder.write("Generated answer.")

        return answer, results

    def get_context(self, results: list) -> str:
        logger.info("Getting context from milvus collection")
        doc_ids = [result['id'] for result in results[:3]]  # Get top 3 results
        docs = self.data_handler.collection.query(
            expr=f"id in {doc_ids}",
            output_fields=["id", "content"]
        )
        documents = {doc['id']: doc['content'] for doc in docs}

        context = ""
        for result in results[:3]:
            if result['id'] in documents:
                context += documents[result['id']] + " "

        return context[:8000] 

    def run(self):
        logger.info("App started")
        st.title("NVIDIA CUDA Documentation QA System")

        if st.button("Crawl and Process Data"):
            self.crawl_and_process()
            st.success("Crawling and processing completed!")

        if st.button("Clear Database"):
            with st.spinner("Clearing database..."):
                self.data_handler.clear_collection()
            st.success("Database cleared successfully!")

        question = st.text_input("Enter your question about CUDA:")

        if st.button("Get Answer"):
            if question:
                answer, results = self.answer_question(question)

                st.subheader("Answer:")
                answer = str(answer).replace('assistant:', '')
                st.write(answer)
                logger.info("Showing answers and relevant documents")
                st.subheader("Top Relevant Documents:")
                for i, result in enumerate(results[:3], 1):
                    with st.expander(f"Document {i}"):
                        st.write(f"Score: {result['rerank_score']:.4f}")
                        doc = self.data_handler.collection.query(
                            expr=f"id == {result['id']}",
                            output_fields=["title", "url"]
                        )[0]
                        st.write(f"Title: {doc['title']}")
                        st.write(f"URL: {doc['url']}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    system = DocQASystem()
    system.run()