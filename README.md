# Semantic Structure-Aware XML Chunker for WSO2 Micro Integrator

A pure parsed-tree, token-driven XML chunking algorithm specifically designed for WSO2 Micro Integrator (MI) configuration files. This tool intelligently breaks down complex XML artifacts into meaningful, context-rich chunks optimized for **RAG (Retrieval-Augmented Generation) systems** and **semantic code retrieval**.

---

## ⚠️ CRITICAL: Understanding Chunk Structure for Semantic Retrieval

> **Chunks are NOT just XML portions!** Each chunk consists of **METADATA as CONTEXT** combined with **Cleaned XML Content**.

### Embedding Text Formula

For semantic retrieval operations, the embedding text for each chunk follows this structure:

```
{ Context Metadata (JSON → Text) } + { Cleaned XML Content }
```

### Why This Matters

1. **Contextual Awareness**: The embedding text always starts with hierarchical context (API name, context path, resource method, URI template), ensuring semantic search understands WHERE in the configuration this chunk belongs.
2. **Semantic Richness**: By converting both metadata and XML structure to natural text, embedding models can capture the full semantic meaning.
3. **Better Retrieval**: When querying "How does BankAPI handle greetings?", the context-prefixed embedding ensures this chunk ranks highly due to both API name and content relevance.

---

## 🎯 Key Features

### 1. **Pure Parsed-Tree Traversal**
Unlike heuristic-based approaches, this algorithm uses pure XML tree traversal. 
- **No Hardcoded Registries**: Works generically across any XML schema without requiring a predefined artifact registry.
- **Dynamic Context**: Context and artifact metadata are read directly from the XML root element's attributes.

### 2. **Token-Driven Boundary Detection**
Every XML tag is a potential chunk boundary.
- **Precise Sizing**: Token count alone decides chunking logic (default 256 tokens for `all-MiniLM-L6-v2`).
- **Real Tokenizer**: Uses the actual Hugging Face tokenizer (`@huggingface/transformers`) to exact-match the embedding model's limits.
- **Auto-Descent**: If a tree node fits within the token limit, it is emitted as a single chunk. If it exceeds the limit, the algorithm seamlessly descends into its children.

### 3. **Context-Enriched Embedding Text** ⭐
- **Context-First Approach**: Each chunk's embedding text starts with structured context metadata.
- **JSON Block Protection**: Intelligently preserves JSON inside `<format>` and `<args>` tags so structured payloads aren't destroyed during cleanup.
- Maintains special characters in XPath expressions (`${}()[]`).

### 4. **Cross-Artifact Reference Tracking**
Automatically detects and tracks references between artifacts within the chunk content:
- `<sequence key="SequenceName"/>` → Sequence references
- `configKey="LocalEntryName"` → Local entry references
- `<endpoint key="EndpointName"/>` → Endpoint references
- `<call-template target="TemplateName"/>` → Template references
- `useConfig="Name"` → Data service config references
- `<call-query href="Name">` → Data service query references

---

## 📊 Chunk Structure

Each generated chunk contains comprehensive metadata:

```typescript
{
  filePath: string;              // Source file path
  chunkType: string;             // Specific tag type (e.g., resource, payloadFactory)
  chunkIndex: number;            // Unique chunk identifier
  startLine: number;             // Start line in source file
  endLine: number;               // End line in source file
  content: string;               // Original raw XML content
  embeddingText: string;         // Cleaned text optimized for embeddings
  context: {                     // Rich hierarchical context
    artifact?: { type, name, ... },
    references?: string[],       // Cross-artifact references
    [key: string]: any           // Dynamic element-level contexts
  },
  sequenceKey?: string;          // For standalone definitions
  isSequenceDefinition?: boolean,// True if chunk is the top-level artifact definition
  referencedSequences?: string[] // All references in chunk
}
```

## � Installation

```bash
npm install
```

## 🚀 Usage

### Basic Usage

```typescript
import { XMLChunker } from './chunker';

const chunker = new XMLChunker();
// Tokenizer is loaded automatically on the first run
const chunks = await chunker.chunkFile('/path/to/artifact.xml');

console.log(`Generated ${chunks.length} chunks`);
chunks.forEach(chunk => {
  console.log(`${chunk.chunkType} at lines ${chunk.startLine}-${chunk.endLine}`);
  console.log(`Embedding: ${chunk.embeddingText.substring(0, 80)}...`);
});
```

### Running Tests

```bash
npm test
```

This comprehensive test suite will:
- Process all XML files in the `artifacts/` folder
- Verify structural chunking capabilities
- Display detailed chunk analysis, statistics, and cross-references
- Validate that all generated embedding texts strictly obey the token limit
- Export generated chunks to `test-output/chunks.json`

## 🔧 Configuration

Edit `config.ts` to adjust settings like tokenizer model and sizing:

```typescript
// Example config.ts structure
export const config = {
  maxTokens: 256,
  tokenizerModel: 'sentence-transformers/all-MiniLM-L6-v2'
};
```

## 📁 Project Structure

```
.
├── chunker.ts              # Main pure-tree chunking algorithm
├── config.ts               # Configuration settings
├── test-chunker.ts         # Comprehensive test suite
├── package.json            # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
├── artifacts/              # Sample WSO2 MI XML files for testing
├── test-output/            # Generated test results (chunks.json)
└── README.md
```

## 🤝 Contributing

To extend the chunking algorithm, you can modify `chunker.ts` directly. Since the architecture no longer relies on a hardcoded artifact registry, most new structural patterns will be automatically supported by the dynamic context extraction and token-driven descent.


## 📚 References

- [WSO2 Micro Integrator Documentation](https://ei.docs.wso2.com/en/latest/micro-integrator/overview/introduction/)
- [fast-xml-parser](https://github.com/NaturalIntelligence/fast-xml-parser)
- [HuggingFace Transformers for Node](https://huggingface.co/docs/transformers.js)
