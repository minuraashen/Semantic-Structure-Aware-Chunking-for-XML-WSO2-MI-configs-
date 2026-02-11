# Semantic Structure-Aware XML Chunker for WSO2 Micro Integrator

A semantic, hierarchical, and size-aware XML chunking algorithm specifically designed for WSO2 Micro Integrator (MI) configuration files. This tool intelligently breaks down complex XML artifacts into meaningful, context-rich chunks optimized for **RAG (Retrieval-Augmented Generation) systems** and **semantic code retrieval**.

---

## ⚠️ CRITICAL: Understanding Chunk Structure for Semantic Retrieval

> **Chunks are NOT just XML portions!** Each chunk consists of **METADATA as CONTEXT** combined with **Cleaned XML Content**.

### Embedding Text Formula

For semantic retrieval operations, the embedding text for each chunk follows this structure:

```
{ Context Metadata (JSON → Text) } + { Cleaned XML Content }
```

### Example

Given this chunk context and XML content:

**Context (from parent traversal):**
```json
{
  "api": {
    "name": "BankAPI",
    "context": "/bankapi"
  }
}
```

**XML Content:**
```xml
<Custom.greet methods="GET" uri-template="/">
    <inSequence>
        <payloadFactory media-type="json">
            <format>{"greetings":"Welcome to O2 Bank !!"}</format>
        </payloadFactory>
        <respond/>
    </inSequence>
    <faultSequence>
    </faultSequence>
</Custom.greet>
```

**Generated Embedding Text:**
```
API: BankAPI Context: /bankapi Custom.greet methods=GET uri-template=/ inSequence payloadFactory media-type=json format greetings Welcome to O2 Bank respond faultSequence
```

### Why This Matters

1. **Contextual Awareness**: The embedding text always starts with hierarchical context (API name, context path, resource method, URI template), ensuring semantic search understands WHERE in the configuration this chunk belongs.

2. **Semantic Richness**: By converting both metadata and XML structure to natural text, embedding models can capture the full semantic meaning.

3. **Better Retrieval**: When querying "How does BankAPI handle greetings?", the context-prefixed embedding ensures this chunk ranks highly due to both API name and content relevance.

---

## 🎯 Key Features

### 1. **Plugin-Based Architecture**
- Extensible artifact detection via registry system
- No hardcoded artifact types
- Easy to add custom plugins for organization-specific artifacts
- Built-in support for 11 WSO2 MI artifact types

### 2. **Semantic Chunking**
- **Intelligent Boundary Detection**: Recognizes semantic boundaries (resources, sequences, mediators)
- **Context-Aware**: Preserves API context, resource methods, URI templates, and sequence information
- **Context-First Embedding**: Each chunk's embedding text begins with structured metadata for better semantic search
- **Semantic Intent Inference**: Automatically categorizes chunks by intent:
  - `validation` - Filter and switch mediators
  - `transformation` - PayloadFactory and enrich operations
  - `delegation` - Call, send, and HTTP operations
  - `response` - Respond mediators
  - `error-handling` - Fault sequences
  - `data-access` - Query and operation definitions
  - `configuration` - Config, property, and trigger definitions

### 3. **Size-Aware Chunking**
- Token-based sizing (default: 256 tokens for all-MiniLM-L6-v2)
- Prevents oversized chunks that exceed model limits
- Top-down hierarchical splitting when chunks are too large
- Configurable token limits

### 4. **Cross-Artifact Reference Tracking**
Automatically detects and tracks references between artifacts:
- `<sequence key="SequenceName"/>` → Sequence references
- `configKey="LocalEntryName"` → Local entry references
- `<endpoint key="EndpointName"/>` → Endpoint references
- `<call-template target="TemplateName"/>` → Template references

### 5. **Content Hashing**
- SHA-256 hashing for each chunk (content + metadata)
- Enables deduplication and change detection
- Useful for incremental updates

### 6. **Context-Enriched Embedding Text** ⭐
- **Context-First Approach**: Each chunk's embedding text starts with structured context metadata
- **Formula**: `{ Metadata Context } + { Cleaned XML Content }` → Single embedding text
- **Format**: `"API: BankAPI Context: /bankapi Method: POST URI: /deposit" + cleaned content`
- Clean, natural text extraction from XML (removes angle brackets and formatting)
- Preserves semantic expressions like `${payload.userId}`
- Maintains special characters in XPath: `${}()[]`
- **Optimized for semantic embeddings**: The combined context + content creates rich, searchable text for vector databases

## 🏗️ Supported WSO2 MI Artifacts

| Artifact Type | Root Tag | Key Features |
|--------------|----------|--------------|
| REST API | `<api>` | Resources, sequences, HTTP methods |
| Proxy Service | `<proxy>` | Targets, endpoints, transport protocols |
| Sequence | `<sequence>` | Mediator chains, filters, switches |
| Endpoint | `<endpoint>` | HTTP addresses, load balancers, failover |
| Local Entry | `<localEntry>` | Reusable configuration entries |
| Template | `<template>` | Reusable sequence/endpoint templates |
| Data Service | `<data>` | Database configs, queries, operations |
| Message Store | `<messageStore>` | JMS, JDBC, RabbitMQ, in-memory stores |
| Message Processor | `<messageProcessor>` | Sampling and forwarding processors |
| Scheduled Task | `<task>` | Cron triggers and task properties |
| Inbound Endpoint | `<inboundEndpoint>` | Protocol-specific inbound configurations |

## 📦 Installation

```bash
npm install
```

## 🚀 Usage

### Basic Usage

```typescript
import { XMLChunker } from './chunker';
import { artifactRegistry } from './artifact-registry';

const chunker = new XMLChunker();
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

This will:
- Process all XML files in the `artifacts/` folder
- Generate detailed chunk analysis
- Display statistics and cross-references
- Export chunks to `test-output/chunks.json`

### Custom Registry

```typescript
import { XMLChunker, ArtifactRegistry } from './chunker';

const customRegistry = new ArtifactRegistry();
// Register custom plugins
customRegistry.registerPlugin(myCustomPlugin);

const chunker = new XMLChunker(null, customRegistry);
```

## 📊 Chunk Structure

Each chunk contains:

```typescript
{
  filePath: string;              // Source file path
  resourceName: string;          // Resource/artifact name
  resourceType: string;          // Artifact type (api, sequence, etc.)
  chunkType: string;             // Specific tag type
  chunkIndex: number;            // Unique chunk identifier
  startLine: number;             // Start line in source file
  endLine: number;               // End line in source file
  content: string;               // Original XML content
  embeddingText: string;         // Cleaned text for embeddings
  semanticType: string;          // Semantic category
  semanticIntent: string;        // Inferred intent
  contentHash: string;           // SHA-256 hash
  context: {                     // Rich context information
    api?: { name, context, xmlns },
    resource?: { method, uriTemplate },
    sequence?: string | { name, xmlns },
    artifact?: { type, name, ... },
    references?: string[]        // Cross-artifact references
  },
  sequenceKey?: string;          // For standalone definitions
  isSequenceDefinition?: boolean,
  referencedSequences?: string[] // All references in file
}
```

## 🔧 Configuration

Edit `config.ts` to adjust settings:

```typescript
export const config = {
  maxTokens: 256,  // Maximum tokens per chunk
  encoding: 'utf-8'
};
```

## 🎨 Example Output

**Input XML:**
```xml
<resource methods="POST" uri-template="/deposit">
  <inSequence>
    <variable name="amount" expression="${payload.amount}" type="DOUBLE"/>
    <payloadFactory media-type="json">
      <format>{"status": "success"}</format>
    </payloadFactory>
    <respond/>
  </inSequence>
</resource>
```

**Generated Chunks:**

1. **Resource Chunk**
   - Type: `resource`
   - Semantic Type: `resource`
   - Intent: `processing`
   - Context: `API: BankAPI, Method: POST, URI: /deposit`
   - Embedding Text: `API: BankAPI Context: /bankapi Method: POST URI: /deposit resource methods=POST uri-template=/deposit ...`
   - Lines: 13-47

2. **Variable Chunk**
   - Type: `variable`
   - Semantic Type: `component`
   - Intent: `processing`
   - Embedding Text: `API: BankAPI Context: /bankapi Method: POST URI: /deposit variable name=amount expression=${payload.amount} type=DOUBLE`
   - Lines: 16-16

3. **PayloadFactory Chunk**
   - Type: `payloadFactory`
   - Semantic Type: `payloadFactory`
   - Intent: `transformation`
   - Embedding Text: `API: BankAPI Context: /bankapi Method: POST URI: /deposit payloadFactory media-type=json format status success`
   - Lines: 34-45

4. **Respond Chunk**
   - Type: `respond`
   - Semantic Type: `response`
   - Intent: `response`
   - Embedding Text: `API: BankAPI Context: /bankapi Method: POST URI: /deposit respond`
   - Lines: 46-46

**Key Insight**: Notice how each chunk's embedding text begins with contextual metadata (`API: BankAPI Context: /bankapi Method: POST URI: /deposit`), providing rich semantic context for better retrieval in RAG systems.

## 🧪 Testing

The test suite verifies:

1. ✅ Artifact Registry functionality
2. ✅ XML file processing and chunk generation
3. ✅ Detailed chunk analysis with statistics
4. ✅ Hierarchical relationships (when present)
5. ✅ Cross-artifact reference detection
6. ✅ Token size verification
7. ✅ JSON export functionality

**Test Results:**
- Processes all XML files in `artifacts/` folder
- Color-coded terminal output
- Detailed chunk information with previews
- Statistics on chunk types and semantic types
- Reference tracking and validation

## 📁 Project Structure

```
.
├── artifact-registry.ts    # Plugin registry and artifact definitions
├── chunker.ts              # Main chunking algorithm
├── config.ts               # Configuration settings
├── content-hash.ts         # Content hashing utilities (SHA-256)
├── test-chunker.ts         # Comprehensive test suite
├── package.json            # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
├── artifacts/              # Sample WSO2 MI XML files
│   ├── apis/
│   ├── sequences/
│   ├── data-services/
│   ├── data-sources/
│   └── local-entries/
├── test-output/            # Generated test results
│   └── chunks.json         # Exported chunks
└── README.md
```

## 🔌 Adding Custom Plugins

Create a custom plugin for organization-specific artifacts:

```typescript
import { ArtifactPlugin } from './artifact-registry';

const myPlugin: ArtifactPlugin = {
  id: 'customArtifact',
  rootTags: ['customRoot'],
  semanticBoundaries: ['section', 'block'],
  mediatorTags: ['customMediator'],
  atomicTags: ['atomic'],
  extractMetadata: (rootTag, attrs) => ({
    type: 'customArtifact',
    name: attrs.name || 'unknown',
    xmlns: attrs.xmlns
  })
};

artifactRegistry.registerPlugin(myPlugin);
```

## 🎯 Primary Use Case: Semantic Code Retrieval for RAG

This chunking algorithm is specifically designed for **semantic code retrieval** in RAG-based applications:

| Use Case | How Chunks Help |
|----------|-----------------|
| **RAG Systems** | Context-prefixed embedding text (`API: X Context: /y ...`) enables precise retrieval |
| **Semantic Search** | Natural language queries match contextual metadata + code content |
| **AI Code Assistants** | Chunks provide complete context for LLMs to understand configuration purpose |
| **Documentation Generation** | Rich metadata enables automatic documentation from configs |
| **Change Detection** | SHA-256 content hashing tracks configuration changes |
| **Dependency Analysis** | Cross-artifact reference tracking reveals relationships |
| **Configuration Validation** | Semantic type/intent classification aids analysis |

## 📈 Performance

- Handles large XML files efficiently
- Streaming-based processing
- Minimal memory footprint
- Fast pattern matching with optimized regex

## 🤝 Contributing

To extend the chunking algorithm:

1. Add new plugins to `BUILTIN_PLUGINS` in `artifact-registry.ts`
2. Update semantic boundaries and mediator tags
3. Enhance metadata extraction logic
4. Add test cases in `test-chunker.ts`

## 📝 License

[Your License Here]

## 🐛 Known Issues

- XML declaration lines (`<?xml...?>`) may be captured as separate chunks in some cases
- Very deeply nested structures may require token limit adjustments

## 📚 References

- [WSO2 Micro Integrator Documentation](https://ei.docs.wso2.com/en/latest/micro-integrator/overview/introduction/)
- [fast-xml-parser](https://github.com/NaturalIntelligence/fast-xml-parser)
