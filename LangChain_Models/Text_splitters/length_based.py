from langchain_text_splitters import TextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Text_splitters/dl-curriculum.pdf')
docs=loader.load()

text = """The lighthouse stood as a silent sentinel on the jagged cliff, its white paint blistered and peeling from decades of salt spray and gale-force winds. For over a century, it had warned sailors away from the treacherous rocks below, its powerful beam cutting through the blackest nights. But tonight, the light was dark.

Elias, the last keeper, had climbed the spiraling stairs for the final time that evening. The automation system had been installed a week ago, a sleek, humming box of circuits that rendered his life’s work obsolete. He ran a gnarled hand over the cold brass of the great lens, a masterpiece of glass and engineering that had been his only companion for forty years. He remembered the storms he’d weathered, the ships he’d guided to safety, the profound silence of a world reduced to just him and the sea.

Below, the ocean churned, a restless, ink-black beast. Its rhythmic roar was the only sound, a constant lullaby and a constant threat. Elias didn't feel obsolete; he felt like a ghost haunting his own life. The sea didn't need him anymore. The ships had their satellites and their GPS. The world had moved on, leaving him and his lighthouse behind.

With a heavy sigh, he descended the stairs, his footsteps echoing in the hollow tower. He stepped outside, the wind whipping at his coat. He took one last look at the dark tower, a monument to a bygone era. Then, he turned his back on the sea and walked toward the faint glow of the mainland, carrying a lifetime of light within him, even as the lighthouse itself went dark."""

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator='')
#splitter = TextSplitter(chunk_size=200, chunk_overlap=50)

result = splitter.split_text(text)
result1 = splitter.split_documents(docs)

print(result)
print(result1[0].page_content)
