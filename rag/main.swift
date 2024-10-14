import Foundation
import NaturalLanguage

class Document {
    let id: String
    let content: String
    var embedding: [Double]?
    
    init(id: String, content: String) {
        self.id = id
        self.content = content
    }
}

class RAGSystem {
    private var documents: [Document] = []
    private let embeddingModel: NLEmbedding
    
    init() {
        guard let model = NLEmbedding.wordEmbedding(for: .english) else {
            fatalError("Unable to load embedding model")
        }
        self.embeddingModel = model
    }
    
    func addDocument(_ document: Document) {
        let words = document.content.components(separatedBy: .whitespacesAndNewlines)
        let embeddings = words.compactMap { embeddingModel.vector(for: $0) }
        let averageEmbedding = average(embeddings)
        document.embedding = averageEmbedding
        documents.append(document)
    }
    
    func searchRelevantDocuments(for query: String, limit: Int = 3) -> [Document] {
        let queryEmbedding = getEmbedding(for: query)
        let sortedDocuments = documents.sorted { doc1, doc2 in
            guard let emb1 = doc1.embedding, let emb2 = doc2.embedding else { return false }
            return cosineSimilarity(queryEmbedding, emb1) > cosineSimilarity(queryEmbedding, emb2)
        }
        return Array(sortedDocuments.prefix(limit))
    }
    
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
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: parameters)
        
        let semaphore = DispatchSemaphore(value: 0)
        var responseText = ""
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }
            
            guard let data = data else {
                print("No data received")
                return
            }
            
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let response = json["response"] as? String {
                responseText = response
            } else {
                print("Failed to parse response")
            }
        }
        
        task.resume()
        semaphore.wait()
        
        return responseText
    }
    
    private func getEmbedding(for text: String) -> [Double] {
        let words = text.components(separatedBy: .whitespacesAndNewlines)
        let embeddings = words.compactMap { embeddingModel.vector(for: $0) }
        return average(embeddings)
    }
    
    private func average(_ vectors: [[Double]]) -> [Double] {
        guard !vectors.isEmpty else { return [] }
        let sum = vectors.reduce(into: Array(repeating: 0.0, count: vectors[0].count)) { result, vector in
            for (index, value) in vector.enumerated() {
                result[index] += value
            }
        }
        return sum.map { $0 / Double(vectors.count) }
    }
    
    private func cosineSimilarity(_ v1: [Double], _ v2: [Double]) -> Double {
        guard v1.count == v2.count else { return 0 }
        let dotProduct = zip(v1, v2).map(*).reduce(0, +)
        let magnitude1 = sqrt(v1.map { $0 * $0 }.reduce(0, +))
        let magnitude2 = sqrt(v2.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (magnitude1 * magnitude2)
    }
}

// Example usage
let ragSystem = RAGSystem()

// Adding documents to the knowledge base
ragSystem.addDocument(Document(id: "1", content: "Swift is a programming language developed by Apple for iOS, macOS, watchOS, and tvOS."))
ragSystem.addDocument(Document(id: "2", content: "Swift was designed to be safer and more concise than Objective-C, with modern features."))
ragSystem.addDocument(Document(id: "3", content: "Key features of Swift include type safety, type inference, and automatic memory management."))

// Generating a response
let query = "What is Swift and what are its main characteristics?"
let response = ragSystem.generateResponse(for: query)
print("Question: \(query)")
print("Answer: \(response)")
