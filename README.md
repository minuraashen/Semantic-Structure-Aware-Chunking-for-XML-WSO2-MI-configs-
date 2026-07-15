# Semantic Structure-Aware XML Chunker for WSO2 Micro Integrator

A structure-aware, token-budgeted XML chunking algorithm for WSO2 Micro Integrator (Synapse) configuration files, designed for **RAG (Retrieval-Augmented Generation)** and **semantic code retrieval**. The algorithm is artifact-agnostic — it works on any XML schema, since it operates purely on the parsed tree.

Comes with an **evaluation harness** (baselines + labeled queries + metrics) that runs fully locally with lightweight embedding models, matching the privacy constraints of on-prem WSO2 MI deployments.

---

## How it works

```
XML source ──> sax parser ──> position-annotated element tree
                                       │
                    token-gated traversal (embedding-model tokenizer)
                                       │
              ┌────────────────────────┼─────────────────────────┐
        fits budget               too large                 tiny siblings
        → emit chunk            → descend into              → aggregate until
                                  children                    minTokens reached
                                       │
                     embeddingText = context prefix + linearized content
```

1. **Position-annotated parse** (`xml-parser.ts`, built on [sax](https://github.com/isaacs/sax-js)). Every element carries its exact character offsets, so each chunk's `content` is an exact source slice with exact `startLine`/`endLine` — structure and content come from one representation, with no re-location step that could disagree with the tree.

2. **Token-gated traversal** (`chunker.ts`). An element whose embedding text fits `maxTokens` (default 256, measured with the embedding model's own tokenizer) becomes a chunk; otherwise the traversal descends into its children. Oversized leaves are split into overlapping token-budgeted parts.

3. **Sibling aggregation.** Consecutive small siblings are buffered until the combined content reaches `minTokens` (default 64) — so one-line elements (`<temperature>0.7</temperature>`) never become standalone noise chunks. Undersized tails merge backward into the preceding sibling chunk. Thresholds are token-based, so behavior does not depend on source formatting.

4. **Context-enriched embedding text.** Every chunk's embedding text starts with a structurally derived context prefix — artifact metadata plus the (capped) ancestor path:

   ```
   api BankAPI context=/bankapi resource methods=POST uri-template=/deposit inSequence
   http.post configKey=CurrencyConverter relativePath /currency/rate ...
   ```

   This is the deterministic, zero-LLM-cost analogue of contextual retrieval: the XML hierarchy already tells us where a chunk belongs.

5. **Content linearization.** Chunk content is converted to natural text directly from the parsed tree: tag names, `attr=value` pairs (entity-decoded, xmlns dropped), and text content in document order. JSON payloads inside `<format>`/`<args>` and CDATA sections are preserved verbatim. Synapse expression characters (`${} [] / . : @`) survive cleanup.

6. **Cross-artifact reference tracking.** References are extracted from the parsed tree (insensitive to attribute order): `<sequence key>`, `configKey`, `<endpoint key>`, `<call-template target>`, `useConfig`, `<call-query href>`.

## Chunk structure

```typescript
{
  filePath: string;            // repo-relative source path
  chunkType: string;           // element tag, or 'aggregated'
  memberTags?: string[];       // tags inside an aggregated chunk
  chunkIndex: number;
  part?: number;               // for split oversized leaves
  startLine: number;           // exact source lines
  endLine: number;
  startOffset: number;         // exact source character span
  endOffset: number;
  content: string;             // exact source slice
  embeddingText: string;       // context prefix + linearized content
  contentHash: string;         // sha256 prefix, for deduplication
  context: {
    artifact: { type, name, ... },   // root element metadata
    path: [{ tag, attrs }, ...],     // ancestor chain (capped)
    references?: string[]
  },
  sequenceKey?: string;        // artifact name for whole-artifact chunks
  isSequenceDefinition: boolean;
  referencedSequences: string[];
}
```

## Usage

```bash
npm install
```

```typescript
import { XMLChunker } from './chunker';

const chunker = new XMLChunker();
const chunks = await chunker.chunkFile('path/to/artifact.xml');

// Ablation switches (used by the evaluation harness):
const noContext = new XMLChunker({ includeContext: false });
const rawXml    = new XMLChunker({ cleanContent: false });
const noAggr    = new XMLChunker({ aggregate: false });
```

### Scripts

| Command | What it does |
|---|---|
| `npm test` | Unit test suite (25 tests: parser round-trips, adversarial XML, token-limit and coverage invariants, aggregation, references, ablations) |
| `npm run demo` | Chunk everything in `artifacts/`, print detailed analysis, export `test-output/chunks.json` |
| `npm run eval` | Retrieval evaluation: our chunker + 3 ablations vs fixed-size / recursive-split / whole-file baselines, on 36 labeled queries, swept across 5 lightweight local embedding models. Writes `eval/results.json` / `eval/results.md` |

## Configuration (`config.ts`)

```typescript
export const config = {
  maxTokens: 256,           // chunk budget (all-MiniLM-L6-v2 max sequence)
  minTokens: 64,            // aggregation threshold (content tokens)
  maxContextAncestors: 4,   // ancestor path cap in the context prefix
  tokenizerModel: 'sentence-transformers/all-MiniLM-L6-v2',
  embeddingModel: 'Xenova/all-MiniLM-L6-v2',   // eval default
};
```

## Evaluation

The harness (`eval/`) compares, on a labeled query set over `artifacts/`:

- **structural (ours)** — full pipeline
- **ablations** — without context prefix / without cleanup / without aggregation
- **baselines** — fixed-size 256-token windows, LangChain-style recursive splitter, whole-file

Metrics: Success@1, Success@5, MRR@10, nDCG@10, and LineRecall@5 (fraction of gold lines covered by the top-5 retrieved chunks — rewards complete answers, penalizes fragmentation). Embedding models swept: all-MiniLM-L6-v2, all-MiniLM-L6-v2-code-search-512, bge-small-en-v1.5, e5-small-v2, gte-small — all ≤33M params, all running fully locally via ONNX.

See `eval/results.md` for the current numbers.

## Project structure

```
.
├── xml-parser.ts           # position-annotated XML tree (sax-based)
├── chunker.ts              # chunking algorithm
├── config.ts               # budgets and model IDs
├── tests/unit-tests.ts     # assertion-based test suite
├── test-chunker.ts         # demo / analysis script
├── eval/
│   ├── run-eval.ts         # multi-model retrieval evaluation
│   ├── baselines.ts        # fixed-size, recursive-split, whole-file
│   ├── queries.ts          # 36 labeled queries with gold line spans
│   └── results.md          # latest results
├── artifacts/              # sample WSO2 MI XML corpus
└── test-output/            # exported chunks.json
```

## References

- [cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree](https://arxiv.org/abs/2506.15655) (EMNLP 2025 Findings) — the closest prior work; split-then-merge tree chunking for code
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM-generated chunk context; our prefix is its deterministic, structural analogue
- [WSO2 Micro Integrator Documentation](https://mi.docs.wso2.com/)
- [sax-js](https://github.com/isaacs/sax-js) · [Transformers.js](https://huggingface.co/docs/transformers.js)
