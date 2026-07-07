import * as fs from 'fs';
import * as crypto from 'crypto';
import { AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';
import {
  parseXML,
  XMLDocument,
  XMLElementNode,
  XMLNode,
  childElements,
  localName,
} from './xml-parser';
import { config } from './config';

/**
 * Semantic, Hierarchical, Structure-Aware XML Chunker for WSO2 MI artifacts
 * (artifact-agnostic: works on any XML schema).
 *
 * Pipeline per file:
 *   1. Parse into a position-annotated element tree (sax-based; exact source
 *      offsets per element — structure and content come from ONE representation).
 *   2. Traverse top-down with a token gate: an element whose embedding text
 *      fits within `maxTokens` becomes a chunk candidate; otherwise descend
 *      into its children.
 *   3. Aggregate consecutive small siblings until the combined embedding text
 *      reaches `minTokens` (token-based, formatting-independent), so one-line
 *      elements never become standalone noise chunks. An undersized tail is
 *      merged backward into the preceding sibling chunk when it fits.
 *   4. Each chunk's embedding text = [ancestor context prefix] + [linearized,
 *      cleaned element content]. Context is derived structurally from the
 *      ancestor path — no LLM calls (cf. contextual retrieval).
 *
 * Token counts always use the embedding model's own tokenizer, so the gate is
 * exact with respect to what the model receives. Within a sibling list the
 * gate is O(n): WordPiece token counts are additive across whitespace-joined
 * segments, so buffered totals are running sums of per-element counts
 * (verified by a unit test) instead of re-encoding the growing buffer.
 */

/** One ancestor on the path from the artifact root down to a chunk. */
export interface AncestorEntry {
  tag: string;
  /** Ancestor attributes (xmlns declarations excluded). Empty for bare wrappers. */
  attrs: Record<string, string>;
}

/**
 * Semantic context attached to every chunk.
 * `artifact` is root-level metadata read from the XML root element.
 * `path` is the ordered ancestor chain (nearest last, capped at
 * `maxContextAncestors`) — an array, so same-named ancestors at different
 * depths never collide.
 */
export interface SemanticContext {
  artifact: {
    type: string;
    name: string;
    [key: string]: string;
  };
  path: AncestorEntry[];
  /** Cross-artifact references found in the chunk content. */
  references?: string[];
}

export interface XMLChunk {
  filePath: string;
  /** Tag name of the chunk's element, or 'aggregated' for merged sibling runs. */
  chunkType: string;
  /** Tags of the merged elements when chunkType === 'aggregated'. */
  memberTags?: string[];
  chunkIndex: number;
  /** 1-based part number when an oversized leaf was split into several chunks. */
  part?: number;
  startLine: number;
  endLine: number;
  /** Exact character span in the source file. */
  startOffset: number;
  endOffset: number;
  /** Raw XML content — the exact source text of the chunk's span. */
  content: string;
  /** Context-prefixed, cleaned text that is embedded. */
  embeddingText: string;
  /** sha256 of content (first 16 hex chars) for deduplication. */
  contentHash: string;
  context: SemanticContext;
  /** Artifact name when this chunk is a whole standalone artifact definition. */
  sequenceKey?: string;
  isSequenceDefinition: boolean;
  referencedSequences: string[];
}

/**
 * Ablation switches (all default true). Used by the evaluation harness to
 * isolate the contribution of each component.
 */
export interface ChunkerOptions {
  /** Prepend the ancestor context prefix to embedding text. */
  includeContext?: boolean;
  /** Linearize + clean XML for embedding text (false: raw XML is embedded). */
  cleanContent?: boolean;
  /** Aggregate small consecutive siblings (false: every fitting element is its own chunk). */
  aggregate?: boolean;
}

interface BufferEntry {
  el: XMLElementNode;
  /** Element's linearized text (no context prefix). */
  text: string;
  /** Token count of `text` without special tokens. */
  tokens: number;
}

interface ChunkState {
  chunkIndex: number;
}

export class XMLChunker {
  private tokenizer: PreTrainedTokenizer | null = null;
  /** Number of special tokens ([CLS], [SEP]) the tokenizer adds per encode. */
  private specialTokenCount = 0;
  private readonly maxTokens: number = config.maxTokens;
  private readonly minTokens: number;
  private readonly maxContextAncestors: number = config.maxContextAncestors;
  private readonly includeContext: boolean;
  private readonly cleanContent: boolean;

  constructor(options: ChunkerOptions = {}) {
    this.includeContext = options.includeContext !== false;
    this.cleanContent = options.cleanContent !== false;
    // aggregate=false ⇒ flush after every element (minTokens 0 disables buffering-up).
    this.minTokens = options.aggregate === false ? 0 : config.minTokens;
  }

  /**
   * Load the embedding model tokenizer (idempotent — only loads once).
   * Always the same model as the embedding pipeline, so the token gate is
   * consistent with what the model actually receives.
   */
  async initialize(): Promise<void> {
    if (!this.tokenizer) {
      this.tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
      this.specialTokenCount = this.tokenizer.encode('').length;
    }
  }

  async chunkFile(filePath: string): Promise<XMLChunk[]> {
    await this.initialize();
    const source = await fs.promises.readFile(filePath, 'utf-8');
    return this.chunkSource(source, filePath);
  }

  /** Chunk XML given as a string (used by tests and the evaluation harness). */
  async chunkText(source: string, filePath = '<memory>'): Promise<XMLChunk[]> {
    await this.initialize();
    return this.chunkSource(source, filePath);
  }

  private chunkSource(source: string, filePath: string): XMLChunk[] {
    const doc = parseXML(source);
    const root = doc.roots.find((n): n is XMLElementNode => n.kind === 'element');
    if (!root) return [];

    const context = this.buildRootContext(root);
    const chunks: XMLChunk[] = [];
    const state: ChunkState = { chunkIndex: 0 };

    const prefix = this.contextPrefix(context);
    const prefixTokens = this.countNoSpecial(prefix);
    const rootText = this.elementText(root, doc);
    const rootTokens = this.countNoSpecial(rootText);

    if (prefixTokens + rootTokens + this.specialTokenCount <= this.maxTokens) {
      // Whole artifact fits in a single chunk.
      this.emitChunk([{ el: root, text: rootText, tokens: rootTokens }], prefix, context, doc, filePath, chunks, state, true);
    } else {
      const children = childElements(root);
      if (children.length > 0) {
        this.processSiblings(children, context, doc, filePath, chunks, state);
      }
      if (chunks.length === 0) {
        // Root has no element children (or none produced chunks) but is oversized.
        this.emitOversizedParts(root, rootText, prefix, prefixTokens, context, doc, filePath, chunks, state);
      }
    }

    return chunks;
  }

  /**
   * Root artifact metadata, read directly from the root element.
   * `name` falls back to `key` (local entries) and then to the tag itself.
   */
  private buildRootContext(root: XMLElementNode): SemanticContext {
    const attrs: Record<string, string> = {};
    for (const [k, v] of Object.entries(root.attrs)) {
      if (!k.startsWith('xmlns')) attrs[k] = v;
    }
    const name = attrs.name || attrs.key || root.tag;
    return {
      artifact: { type: root.tag, name, ...attrs },
      path: [],
    };
  }

  /**
   * Token-gated traversal of one sibling list with aggregation.
   *
   * - Element fits → buffered; buffer flushes when it reaches `minTokens`
   *   (or would exceed `maxTokens` by absorbing the next element).
   * - Element too large → flush buffer, descend into its children.
   * - Undersized tail buffer → merged backward into the preceding sibling
   *   chunk when the combined embedding text still fits (span stays
   *   contiguous because both come from the same sibling run).
   */
  private processSiblings(
    els: XMLElementNode[],
    context: SemanticContext,
    doc: XMLDocument,
    filePath: string,
    chunks: XMLChunk[],
    state: ChunkState
  ): void {
    const prefix = this.contextPrefix(context);
    const prefixTokens = this.countNoSpecial(prefix);

    let buffer: BufferEntry[] = [];
    let bufferTokens = 0;
    // Last chunk emitted at THIS level — backward-merge target for a small
    // tail. Reset to null after descending, so merges never span an element
    // that was chunked at a deeper level.
    let lastChunk: XMLChunk | null = null;

    // Returns the emitted chunk (or null if the buffer was empty) so callers
    // can track it — TypeScript's flow analysis ignores closure assignments.
    const flush = (): XMLChunk | null => {
      if (buffer.length === 0) return null;
      const emitted = this.emitChunk(buffer, prefix, context, doc, filePath, chunks, state, false);
      buffer = [];
      bufferTokens = 0;
      return emitted;
    };

    for (const el of els) {
      const text = this.elementText(el, doc);
      const tokens = this.countNoSpecial(text);

      if (prefixTokens + tokens + this.specialTokenCount > this.maxTokens) {
        // Oversized element: flush pending buffer, then descend.
        flush();
        const children = childElements(el);
        if (children.length > 0) {
          const childContext = this.extendContext(context, el);
          const before = chunks.length;
          this.processSiblings(children, childContext, doc, filePath, chunks, state);
          if (chunks.length === before) {
            this.emitOversizedParts(el, text, prefix, prefixTokens, context, doc, filePath, chunks, state);
          }
        } else {
          // Oversized leaf (huge text/attributes, no element children).
          this.emitOversizedParts(el, text, prefix, prefixTokens, context, doc, filePath, chunks, state);
        }
        lastChunk = null;
        continue;
      }

      if (buffer.length > 0 && prefixTokens + bufferTokens + tokens + this.specialTokenCount > this.maxTokens) {
        // Adding this element would exceed the budget — flush what we have.
        const emitted = flush();
        if (emitted) lastChunk = emitted;
      }

      buffer.push({ el, text, tokens });
      bufferTokens += tokens;

      // Content-token threshold (prefix excluded): a chunk must carry at
      // least minTokens of content signal, regardless of prefix length.
      if (bufferTokens >= this.minTokens) {
        const emitted = flush();
        if (emitted) lastChunk = emitted;
      }
    }

    // Tail: an undersized leftover buffer merges backward when possible.
    if (buffer.length > 0) {
      if (bufferTokens < this.minTokens && lastChunk !== null) {
        const tailText = buffer.map((b) => b.text).filter(Boolean).join(' ');
        const candidate = tailText ? `${lastChunk.embeddingText} ${tailText}` : lastChunk.embeddingText;
        if (this.countTokens(candidate) <= this.maxTokens) {
          this.mergeIntoChunk(lastChunk, buffer, candidate, doc);
          return;
        }
      }
      flush();
    }
  }

  /** Emit one chunk from a buffered run of sibling elements. */
  private emitChunk(
    buffer: BufferEntry[],
    prefix: string,
    context: SemanticContext,
    doc: XMLDocument,
    filePath: string,
    chunks: XMLChunk[],
    state: ChunkState,
    isRootChunk: boolean
  ): XMLChunk {
    const first = buffer[0].el;
    const last = buffer[buffer.length - 1].el;
    const content = doc.source.slice(first.startOffset, last.endOffset);
    const bodyText = buffer.map((b) => b.text).filter(Boolean).join(' ');
    const embeddingText = [prefix, bodyText].filter(Boolean).join(' ');

    const refs = new Set<string>();
    for (const entry of buffer) {
      for (const r of this.extractReferences(entry.el)) refs.add(r);
    }
    const references = Array.from(refs);

    const aggregated = buffer.length > 1;
    const chunk: XMLChunk = {
      filePath,
      chunkType: aggregated ? 'aggregated' : first.tag,
      ...(aggregated ? { memberTags: buffer.map((b) => b.el.tag) } : {}),
      chunkIndex: state.chunkIndex++,
      startLine: doc.lineOf(first.startOffset),
      endLine: doc.lineOf(last.endOffset - 1),
      startOffset: first.startOffset,
      endOffset: last.endOffset,
      content,
      embeddingText,
      contentHash: this.hashContent(content),
      context: {
        artifact: context.artifact,
        path: context.path,
        ...(references.length > 0 ? { references } : {}),
      },
      ...(isRootChunk ? { sequenceKey: context.artifact.name } : {}),
      isSequenceDefinition: isRootChunk,
      referencedSequences: references,
    };

    chunks.push(chunk);
    return chunk;
  }

  /**
   * Merge an undersized tail buffer backward into the preceding sibling
   * chunk. The merged span is contiguous source text (same sibling run), so
   * `content` remains an exact source slice.
   */
  private mergeIntoChunk(
    target: XMLChunk,
    buffer: BufferEntry[],
    mergedEmbeddingText: string,
    doc: XMLDocument
  ): void {
    const last = buffer[buffer.length - 1].el;
    target.memberTags = [
      ...(target.memberTags ?? [target.chunkType]),
      ...buffer.map((b) => b.el.tag),
    ];
    target.chunkType = 'aggregated';
    target.endOffset = last.endOffset;
    target.endLine = doc.lineOf(last.endOffset - 1);
    target.content = doc.source.slice(target.startOffset, last.endOffset);
    target.embeddingText = mergedEmbeddingText;
    target.contentHash = this.hashContent(target.content);

    const refs = new Set<string>(target.referencedSequences);
    for (const entry of buffer) {
      for (const r of this.extractReferences(entry.el)) refs.add(r);
    }
    const references = Array.from(refs);
    target.referencedSequences = references;
    if (references.length > 0) target.context.references = references;
  }

  /**
   * An oversized element that cannot be descended into (leaf, or descent
   * produced nothing) is split into multiple chunks by windowing its
   * linearized text, each part staying within the token budget. All parts
   * share the element's source span; `part` disambiguates them.
   */
  private emitOversizedParts(
    el: XMLElementNode,
    text: string,
    prefix: string,
    prefixTokens: number,
    context: SemanticContext,
    doc: XMLDocument,
    filePath: string,
    chunks: XMLChunk[],
    state: ChunkState
  ): void {
    // Reserve room for the prefix; if a deep prefix leaves too little, drop it.
    let effectivePrefix = prefix;
    let budget = this.maxTokens - prefixTokens - this.specialTokenCount;
    if (budget < 16) {
      effectivePrefix = '';
      budget = this.maxTokens - this.specialTokenCount;
    }

    const words = text.split(' ').filter(Boolean);
    const wordTokens = words.map((w) => this.countNoSpecial(w));

    // Greedy word windows with a small overlap so no phrase is cut without
    // context on either side.
    const OVERLAP_WORDS = 10;
    const windows: string[] = [];
    let start = 0;
    while (start < words.length) {
      let used = 0;
      let end = start;
      while (end < words.length && used + wordTokens[end] <= budget) {
        used += wordTokens[end];
        end++;
      }
      if (end === start) {
        // Single word larger than the budget (e.g. a giant protected JSON
        // token) — emit it alone; the embedding model truncates the excess.
        end = start + 1;
      }
      windows.push(words.slice(start, end).join(' '));
      if (end >= words.length) break;
      start = Math.max(end - OVERLAP_WORDS, start + 1);
    }

    const content = doc.source.slice(el.startOffset, el.endOffset);
    const references = this.extractReferences(el);
    const startLine = doc.lineOf(el.startOffset);
    const endLine = doc.lineOf(el.endOffset - 1);

    windows.forEach((window, i) => {
      chunks.push({
        filePath,
        chunkType: el.tag,
        chunkIndex: state.chunkIndex++,
        ...(windows.length > 1 ? { part: i + 1 } : {}),
        startLine,
        endLine,
        startOffset: el.startOffset,
        endOffset: el.endOffset,
        content,
        embeddingText: [effectivePrefix, window].filter(Boolean).join(' '),
        contentHash: this.hashContent(content),
        context: {
          artifact: context.artifact,
          path: context.path,
          ...(references.length > 0 ? { references } : {}),
        },
        isSequenceDefinition: false,
        referencedSequences: references,
      });
    });
  }

  /**
   * Extend the ancestor path when descending into an element.
   * Path entries are ordered (nearest ancestor last) and capped, so deep
   * nesting cannot crowd chunk content out of the token budget.
   */
  private extendContext(context: SemanticContext, el: XMLElementNode): SemanticContext {
    const attrs: Record<string, string> = {};
    for (const [k, v] of Object.entries(el.attrs)) {
      if (!k.startsWith('xmlns')) attrs[k] = v;
    }
    const path = [...context.path, { tag: el.tag, attrs }];
    return {
      artifact: context.artifact,
      path: path.length > this.maxContextAncestors ? path.slice(-this.maxContextAncestors) : path,
    };
  }

  /**
   * Format the context as a natural-text prefix for embedding.
   * Example: "api BankAPI context=/bankapi resource methods=POST uri-template=/deposit inSequence"
   * Attribute-less wrappers contribute just their tag name (no "x: x" noise).
   */
  private contextPrefix(context: SemanticContext): string {
    if (!this.includeContext) return '';

    const parts: string[] = [];
    const { type, name, ...rest } = context.artifact;
    parts.push(type);
    if (name && name !== type) parts.push(name);
    for (const [k, v] of Object.entries(rest)) {
      if (v !== undefined && v !== null && v !== '' && k !== 'name' && k !== 'key') {
        parts.push(`${k}=${v}`);
      }
    }

    for (const entry of context.path) {
      parts.push(entry.tag);
      for (const [k, v] of Object.entries(entry.attrs)) {
        if (v !== '') parts.push(`${k}=${v}`);
      }
    }

    return parts.join(' ');
  }

  /**
   * The element's contribution to embedding text.
   * cleanContent=true (default): tree linearization (see linearizeElement).
   * cleanContent=false (ablation): whitespace-normalized raw source.
   */
  private elementText(el: XMLElementNode, doc: XMLDocument): string {
    if (!this.cleanContent) {
      return doc.source.slice(el.startOffset, el.endOffset).replace(/\s+/g, ' ').trim();
    }
    return this.linearizeElement(el);
  }

  /**
   * Linearize an element subtree into cleaned natural text directly from the
   * parsed tree (no regex over raw XML): tag names, attribute name=value
   * pairs (entity-decoded, xmlns dropped), and text content in document
   * order. JSON payloads inside <format>/<args> and CDATA sections are
   * preserved verbatim so structured payloads survive cleanup.
   */
  private linearizeElement(el: XMLElementNode): string {
    const segments: Array<{ text: string; protect: boolean }> = [];

    const visit = (node: XMLNode): void => {
      if (node.kind === 'text') {
        const trimmed = node.text.trim();
        if (trimmed) segments.push({ text: trimmed, protect: node.cdata });
        return;
      }

      segments.push({ text: node.tag, protect: false });
      for (const [k, v] of Object.entries(node.attrs)) {
        if (k.startsWith('xmlns')) continue;
        segments.push({ text: v === '' ? k : `${k}=${v}`, protect: false });
      }

      // JSON payload protection: keep structured payloads intact.
      const local = localName(node.tag);
      if (local === 'format' || local === 'args') {
        const inner = node.children
          .filter((c): c is Extract<XMLNode, { kind: 'text' }> => c.kind === 'text')
          .map((c) => c.text)
          .join(' ')
          .trim();
        if (inner.startsWith('{') || inner.startsWith('[')) {
          segments.push({ text: inner.replace(/\s+/g, ' '), protect: true });
          return; // do not recurse — the payload was consumed verbatim
        }
      }

      for (const child of node.children) visit(child);
    };

    visit(el);

    const cleaned = segments
      .map((s) => (s.protect ? s.text.replace(/\s+/g, ' ').trim() : this.cleanSegment(s.text)))
      .filter(Boolean);

    return cleaned.join(' ');
  }

  /**
   * Clean one unprotected text segment: strip symbols that carry no
   * embedding signal while preserving $ { } [ ] / - . , : @ used in synapse
   * expressions and paths; then drop 1-char non-numeric fragments.
   */
  private cleanSegment(s: string): string {
    return s
      .replace(/[^\w\s=\$\{\}\[\]\/\-\.,:@]/g, ' ')
      .split(/\s+/)
      .filter((t) => (t.length > 1 || /^\d+$/.test(t)) && t.length < 100)
      .join(' ');
  }

  /**
   * Cross-artifact references, extracted from the parsed tree — insensitive
   * to attribute order and tag formatting (the previous regex approach
   * required e.g. `key` to be the first attribute).
   */
  private extractReferences(el: XMLElementNode): string[] {
    const refs = new Set<string>();

    const visit = (node: XMLElementNode): void => {
      const local = localName(node.tag);
      const a = node.attrs;
      if (local === 'sequence' && a.key) refs.add(`sequence:${a.key}`);
      if (a.configKey) refs.add(`localEntry:${a.configKey}`);
      if (local === 'endpoint' && a.key) refs.add(`endpoint:${a.key}`);
      if (local === 'call-template' && a.target) refs.add(`template:${a.target}`);
      if (a.useConfig) refs.add(`config:${a.useConfig}`);
      if (local === 'call-query' && a.href) refs.add(`query:${a.href}`);
      for (const child of childElements(node)) visit(child);
    };

    visit(el);
    return Array.from(refs);
  }

  /** Exact token count including special tokens — what the model receives. */
  private countTokens(text: string): number {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized. Call initialize() before counting tokens.');
    }
    return this.tokenizer.encode(text).length;
  }

  /**
   * Token count without special tokens. Additive across whitespace-joined
   * segments for WordPiece tokenizers (asserted in unit tests), which makes
   * sibling-buffer gating O(n) instead of re-encoding the growing buffer.
   */
  private countNoSpecial(text: string): number {
    if (text === '') return 0;
    return this.countTokens(text) - this.specialTokenCount;
  }

  private hashContent(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex').slice(0, 16);
  }
}
