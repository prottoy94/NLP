from langchain_community.document_loaders import  PyPDFLoader
loader = PyPDFLoader("Document_loaders/dl-curriculum.pdf")
docs = loader.load()
print(docs)
print(len(docs))
print(docs[0])
print(docs[0].page_content)
print(docs[0].metadata)