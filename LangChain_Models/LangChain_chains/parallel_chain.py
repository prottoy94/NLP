import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq   
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )

model1 = ChatGroq(
    model="llama-3.1-8b-instant",  # Active model
    temperature=0.9,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt1=PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate a 5 short question and answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
    input_variables=["notes", "quiz"]
)

parser=StrOutputParser()

parallel_chain= RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model | parser
})

merge_chain= prompt3 | model | parser

chain= parallel_chain | merge_chain
text="""Deep Learning is transforming the way machines understand, learn and interact with complex data. Deep learning mimics neural networks of the human brain, it enables computers to autonomously uncover patterns and make informed decisions from vast amounts of unstructured data.

deep_learning_process.webpdeep_learning_process.webp
Working of Deep Learning
Neural network consists of layers of interconnected nodes or neurons that collaborate to process input data. In a fully connected deep neural network data flows through multiple layers where each neuron performs nonlinear transformations, allowing the model to learn intricate representations of the data.

In a deep neural network the input layer receives data which passes through hidden layers that transform the data using nonlinear functions. The final output layer generates the model’s prediction.
Machine Learning vs Deep Learning
Machine learning and Deep Learning both are subsets of artificial intelligence but there are many similarities and differences between them.
Evolution of Neural Architectures
Perceptron (1950s)

First simple neural network with a single layer
Could only solve linearly separable problems
Failed on complex tasks like the XOR problem
Multi-Layer Perceptrons (MLPs)

Introduced hidden layers and non-linear activation functions
Enabled modeling of non-linear relationships
Trained effectively using backpropagation
Marked a major leap in neural network capabilities
Types of neural networks
Feedforward neural networks (FNNs): They are the simplest type of ANN, where data flows in one direction from input to output. It is used for basic tasks like classification.
Convolutional Neural Networks (CNNs): They are specialized for processing grid-like data, such as images. CNNs use convolutional layers to detect spatial hierarchies, making them ideal for computer vision tasks.
Recurrent Neural Networks (RNNs): Theyare able to process sequential data, such as time series and natural language. RNNs have loops to retain information over time, enabling applications like language modeling and speech recognition. Variants like LSTMs and GRUs address vanishing gradient issues.
Generative Adversarial Networks (GANs): This consist of two networks—a generator and a discriminator—that compete to create realistic data. GANs are widely used for image generation, style transfer and data augmentation.
Autoencoders: They are unsupervised networks that learn efficient data encodings. They compress input data into a latent representation and reconstruct it, useful for dimensionality reduction and anomaly detection.
Transformer Networks: It has revolutionized NLP with self-attention mechanisms. Transformers excel at tasks like translation, text generation and sentiment analysis, powering models like GPT and BERT.
Applications
1. Computer vision
In computer vision, deep learning models enable machines to identify and understand visual data. Some of the main applications of deep learning in computer vision include:

Object detection and recognition: Deep learning models are used to identify and locate objects within images and videos, making it possible for machines to perform tasks such as self-driving cars, surveillance and robotics. 
Image classification: Deep learning models can be used to classify images into categories such as animals, plants and buildings. This is used in applications such as medical imaging, quality control and image retrieval. 
Image segmentation: Deep learning models can be used for image segmentation into different regions, making it possible to identify specific features within images.
2. Natural language processing (NLP)
In NLP, deep learning model enable machines to understand and generate human language. Some of the main applications of deep learning in NLP include: 

Automatic Text Generation: Deep learning model can learn the corpus of text and new text like summaries, essays can be automatically generated using these trained models.
Language translation: Deep learning models can translate text from one language to another, making it possible to communicate with people from different linguistic backgrounds. 
Sentiment analysis: Deep learning models can analyze the sentiment of a piece of text, making it possible to determine whether the text is positive, negative or neutral.
Speech recognition: Deep learning models can recognize and transcribe spoken words, making it possible to perform tasks such as speech-to-text conversion, voice search and voice-controlled devices. 
3. Reinforcement learning
In reinforcement learning, deep learning works as training agents to take action in an environment to maximize a reward. Some of the main applications of deep learning in reinforcement learning include: 

Game playing: Deep reinforcement learning models have been able to beat human experts at games such as Go, Chess and Atari. 
Robotics: Deep reinforcement learning models can be used to train robots to perform complex tasks such as grasping objects, navigation and manipulation. 
Control systems: Deep reinforcement learning models can be used to control complex systems such as power grids, traffic management and supply chain optimization. 
Advantages
Deep learning algorithms can achieve very high accuracy in tasks like image recognition and natural language processing.
They can automatically learn important features from data without the need for manual feature engineering.
These models can scale well to handle large and complex datasets, learning from massive amounts of data.
They are flexible and can be applied to a wide range of tasks involving different types of data such as images, text and speech.
Disadvantages
Deep learning requires large amounts of data for effective training, which can be difficult to collect.
Training these models is computationally expensive and often requires specialized hardware like GPUs and TPUs.
The models are complex and behave like a black box, making their results difficult to interpret.
If trained excessively on the same data, they can become too specialized, leading to overfitting and poor performance on new data."""

chain_result=chain.invoke({"text":text})
print(chain_result)
chain.get_graph().print_ascii()