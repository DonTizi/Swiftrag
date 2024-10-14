# RAG in Swift with Local LLM Integration

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Code Explanation](#code-explanation)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Introduction

This project implements a Retrieval Augmented Generation (RAG) system in Swift, designed to work with local Large Language Models (LLMs) like Ollama. It enables context-aware AI responses in iOS and macOS applications by combining document retrieval with natural language generation.

## Prerequisites

- Xcode 12.0 or later
- iOS 14.0+ / macOS 11.0+
- [Ollama](https://ollama.ai/) installed on your local machine

## Setup

1. Clone this repository
2. Ensure Ollama is running on your local machine
3. Add the Swift files to your Xcode project

## Code Explanation

### Main Components

1. `Document`: Represents a single document in the knowledge base
2. `RAGSystem`: The core class that handles document storage, retrieval, and response generation

### Key Methods

```swift
class RAGSystem {
    func addDocument(_ document: Document)
    func searchRelevantDocuments(for query: String, limit: Int = 3) -> [Document]
    func generateResponse(for query: String) -> String
    private func callOllama(with prompt: String) -> String
}
```

## How It Works

1. **Document Embedding**: When a document is added, it's converted into a numerical representation (embedding) using Apple's NLP framework.

```swift
func addDocument(_ document: Document) {
    let words = document.content.components(separatedBy: .whitespacesAndNewlines)
    let embeddings = words.compactMap { embeddingModel.vector(for: $0) }
    let averageEmbedding = average(embeddings)
    document.embedding = averageEmbedding
    documents.append(document)
}
```

2. **Relevant Document Retrieval**: When a query is received, the system finds the most relevant documents using cosine similarity.

```swift
func searchRelevantDocuments(for query: String, limit: Int = 3) -> [Document] {
    let queryEmbedding = getEmbedding(for: query)
    let sortedDocuments = documents.sorted { doc1, doc2 in
        guard let emb1 = doc1.embedding, let emb2 = doc2.embedding else { return false }
        return cosineSimilarity(queryEmbedding, emb1) > cosineSimilarity(queryEmbedding, emb2)
    }
    return Array(sortedDocuments.prefix(limit))
}
```

3. **Response Generation**: The system uses the retrieved documents as context and sends a prompt to Ollama for response generation.

```swift
func generateResponse(for query: String) -> String {
    let relevantDocs = searchRelevantDocuments(for: query)
    let context = relevantDocs.map { $0.content }.joined(separator: " ")
    let prompt = """
    Context: \(context)
    Human: \(query)
    Assistant: Based on the given context, I will provide a concise and accurate answer to the question.
    """
    return callOllama(with: prompt)
}
```

4. **Ollama Integration**: The system communicates with the local Ollama instance via HTTP requests.

```swift
private func callOllama(with prompt: String) -> String {
    let ollamaURL = URL(string: "http://localhost:11434/api/generate")!
    var request = URLRequest(url: ollamaURL)
    request.httpMethod = "POST"
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let parameters: [String: Any] = [
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": false
    ]
    
    // HTTP request and response handling...
}
```

## Usage

Here's a basic example of how to use the RAG system:

```swift
let ragSystem = RAGSystem()

// Add documents to the knowledge base
ragSystem.addDocument(Document(id: "1", content: "Swift is a programming language developed by Apple."))
ragSystem.addDocument(Document(id: "2", content: "Swift is designed to be safe, fast, and expressive."))

// Generate a response
let query = "What is Swift and why was it created?"
let response = ragSystem.generateResponse(for: query)
print(response)
```

## Customization

- **Embedding Model**: The system uses Apple's `NLEmbedding`. You can experiment with other embedding techniques for potentially better results.
- **Similarity Metric**: Currently using cosine similarity. You might try other metrics like Euclidean distance.
- **Ollama Model**: The code uses "llama3.2:3b". You can change this to any other model available in your Ollama installation.

## Troubleshooting

1. **Ollama Connection Issues**: 
   - Ensure Ollama is running (`ollama run llama3.2:3b`)
   - Check if the URL and port (11434) are correct for your setup
2. **Slow Response Times**: 
   - Consider reducing the document limit in `searchRelevantDocuments`
   - Use a smaller Ollama model if available
3. **Out of Memory Errors**: 
   - Implement pagination or limit the number of documents stored in memory

Remember to handle errors gracefully in a production environment, especially network-related issues when calling Ollama.

---

This project demonstrates the power of combining Swift's native capabilities with local LLMs. Feel free to contribute, report issues, or suggest improvements!
