import * as fs from 'fs';
import { XMLParser } from 'fast-xml-parser';
import { AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';
import { config } from './config';

/**
 * Semantic, Hierarchical, Structure-Aware XML Chunker for WSO2 MI artifacts
 *
 * Pure parsed-tree traversal — no external registry or heuristic rules.
 * Token count alone drives chunk boundaries; artifact metadata is read
 * directly from the XML root element's attributes.
 */

export interface XMLChunk {
  filePath: string;
  chunkType: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  content: string;
  embeddingText: string;
  contentHash?: string;
  context: SemanticContext;
  sequenceKey?: string;
  isSequenceDefinition?: boolean;
  referencedSequences?: string[];
}

/**
 * Semantic context — fully generic, schema-agnostic.
 *
 * DESIGN: Only two explicit fields exist:
 *   - `artifact`: Root-level artifact metadata (read from the XML root element)
 *   - `references`: Cross-artifact references extracted from content
 *
 * All other context is stored dynamically via the `[key: string]: any` index
 * signature — making the chunker work identically for any XML schema.
 */
export interface SemanticContext {
  // Root-level artifact metadata (always present)
  artifact?: {
    type: string;
    name: string;
    xmlns?: string;
    [key: string]: any;
  };
  // Cross-artifact references extracted from chunk content
  references?: string[];
  // DYNAMIC: All element-level contexts are stored here automatically
  // Examples: { resource: { method: 'GET', uriTemplate: '/' }, filter: { source: '...' } }
  [key: string]: any;
}


interface LineRange {
  start: number;
  end: number;
}

/**
 * Holds accumulated small sibling nodes pending aggregation.
 */
interface AggregationBuffer {
  contents: string[];      // raw XML content strings of each buffered element
  startLine: number;       // line where the first element begins
  endLine: number;         // line where the last element ends
  lineCount: number;       // total lines across all buffered elements
  tokenCount: number;      // token count of the combined embeddingText so far
  references: string[];    // union of all cross-artifact references
  contextPrefix: string;   // formatMetadata(context) — computed once, reused for all
  isAllIsolatedFragments: boolean; // True if every element in the buffer is an isolated fragment
}

export class XMLChunker {
  private chunkCounter = 0;
  private lastSearchPosition: number = 0;
  private readonly maxTokens: number;
  private tokenizer: PreTrainedTokenizer | null = null;

  /**
   * Minimum line span for a chunk to be emitted immediately without aggregation.
   * Elements with fewer than MIN_CHUNK_LINES lines are candidates for sibling merging.
   */
  private readonly MIN_CHUNK_LINES = 5;

  constructor() {
    this.maxTokens = config.maxTokens;
  }

  /**
   * Load the embedding model tokenizer (idempotent — only loads once).
   * Always uses the same model as the embedding pipeline so the token gate
   * is consistent with what the model actually receives.
   */
  async initialize(): Promise<void> {
    if (!this.tokenizer) {
      this.tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
    }
  }

  async chunkFile(filePath: string): Promise<XMLChunk[]> {
    await this.initialize();
    this.chunkCounter = 0;
    this.lastSearchPosition = 0;
    const xmlContent = await fs.promises.readFile(filePath, 'utf-8');
    const lines = xmlContent.split('\n');

    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '',
      removeNSPrefix: false, // Must preserve namespace for accurate heuristics (e.g., wsp:Policy)
      preserveOrder: true,
      alwaysCreateTextNode: false,
    });

    const parsed = parser.parse(xmlContent);
    const chunks: XMLChunk[] = [];

    // Build root context from the parsed tree
    const rootContext = this.buildRootContext(parsed);

    this.processNode(parsed, lines, filePath, chunks, rootContext);

    return chunks;
  }

  /**
   * Build root context directly from the parsed XML tree.
   * Reads the first real root element and captures its tag name + all attributes.
   * No registry — the tree already has everything we need.
   */
  private buildRootContext(parsed: any): SemanticContext {
    const context: SemanticContext = {};

    if (!Array.isArray(parsed)) {
      context.artifact = { type: 'unknown', name: 'unknown' };
      return context;
    }

    // Find the first real element (skip ?xml processing instructions)
    const rootItem = parsed.find(item => {
      const key = Object.keys(item).find(k => k !== ':@');
      return key && !key.startsWith('?');
    });

    if (rootItem) {
      const rootTag = Object.keys(rootItem).find(k => k !== ':@') || 'unknown';
      const rootAttrs = this.extractAllAttributes(rootItem[':@'] || {});
      const name = rootAttrs.name || rootAttrs.key || rootTag;
      context.artifact = { type: rootTag, name, ...rootAttrs };
    } else {
      context.artifact = { type: 'unknown', name: 'unknown' };
    }

    return context;
  }



  /**
   * Extract cross-artifact references from a chunk's XML content.
   * Detects: sequence key, configKey (local entries), endpoint key,
   *          call-template target, useConfig (data service), call-query href.
   */
  private extractReferencesFromContent(content: string): string[] {
    const refs = new Set<string>();
    let match;

    // <sequence key="Name"/> → sequence reference
    const sequenceRefPattern = /<sequence\s+key=["']([^"']+)["']\s*\/>/g;
    while ((match = sequenceRefPattern.exec(content)) !== null) {
      refs.add(`sequence:${match[1]}`);
    }

    // configKey="Name" → local entry reference (used by http.post, email.send, etc.)
    const configKeyPattern = /configKey=["']([^"']+)["']/g;
    while ((match = configKeyPattern.exec(content)) !== null) {
      refs.add(`localEntry:${match[1]}`);
    }

    // <endpoint key="Name"/> → endpoint reference
    const endpointRefPattern = /<endpoint\s+key=["']([^"']+)["']\s*\/>/g;
    while ((match = endpointRefPattern.exec(content)) !== null) {
      refs.add(`endpoint:${match[1]}`);
    }

    // <call-template target="Name"/> → template reference
    const templateRefPattern = /<call-template\s+target=["']([^"']+)["']/g;
    while ((match = templateRefPattern.exec(content)) !== null) {
      refs.add(`template:${match[1]}`);
    }

    // useConfig="Name" → data service config reference
    const useConfigPattern = /useConfig=["']([^"']+)["']/g;
    while ((match = useConfigPattern.exec(content)) !== null) {
      refs.add(`config:${match[1]}`);
    }

    // <call-query href="Name"> → data service query reference
    const callQueryPattern = /<call-query\s+href=["']([^"']+)["']/g;
    while ((match = callQueryPattern.exec(content)) !== null) {
      refs.add(`query:${match[1]}`);
    }

    return Array.from(refs);
  }

  /**
   * Identifies elements that lack standalone semantic weight (zero-attribute action mediators
   * or empty property resets) to prevent them from being emitted as isolated standalone chunks.
   */
  private isIsolatedFragment(element: any, attrs: Record<string, string>): boolean {
    // fast-xml-parser with preserveOrder=true parses tag contents as arrays.
    // An empty array [] means no children and no text.
    if (Array.isArray(element) && element.length > 0) return false;
    if (typeof element === 'string' && element.trim() !== '') return false;
    if (typeof element === 'object' && element !== null && !Array.isArray(element)) {
      const childKeys = Object.keys(element).filter(k => k !== ':@');
      if (childKeys.length > 0) return false;
    }

    const cleanAttrs = this.extractAllAttributes(attrs);
    const attrKeys = Object.keys(cleanAttrs);

    if (attrKeys.length === 0) return true;
    if (cleanAttrs.action === 'remove') return true;

    const hasValueOrExpr = (cleanAttrs.value !== undefined && cleanAttrs.value !== '') ||
                           (cleanAttrs.expression !== undefined && cleanAttrs.expression !== '');
    const hasReference = cleanAttrs.key !== undefined || cleanAttrs.href !== undefined ||
                         cleanAttrs.target !== undefined || cleanAttrs.sequence !== undefined;

    // It's isolated if it acts as a declaration/reset without substantive payload/reference
    if (!hasValueOrExpr && !hasReference) return true;

    return false;
  }

  /**
   * PURE TREE TRAVERSAL with token gating + sibling aggregation
   *
   * Every XML tag is a potential chunk boundary — no heuristics, no registry rules.
   * Token count alone decides: fits → chunk, too big → descend into children.
   *
   * Consecutive siblings that fit the token limit are buffered together and emitted
   * as a single aggregated chunk once the combined line count reaches MIN_CHUNK_LINES.
   * This means a small <property> (1 line) CAN be co-aggregated with a larger sibling
   * like <call-template> (5 lines) when they are next to each other — the buffer fires
   * exactly when the combined lines first reach the threshold.
   *
   * Context metadata is prepended only once per aggregated group since all siblings
   * share the same parent context.
   */
  private processNode(
    node: any,
    lines: string[],
    filePath: string,
    chunks: XMLChunk[],
    context: SemanticContext
  ): void {
    if (!Array.isArray(node)) return;

    // Aggregation buffer: accumulates consecutive fit siblings until threshold is met
    let buffer: AggregationBuffer | null = null;
    let lastSiblingChunk: XMLChunk | null = null;

    // Compute the context prefix once — all siblings share the same parent context,
    // so we only need this once and reuse it for every buffer token-count check.
    const contextPrefix = this.formatMetadata(context);

    for (const item of node) {
      const tagName = Object.keys(item).find(key => key !== ':@') || '';
      if (!tagName) continue;

      // Skip XML declaration, processing instructions, and #text pseudo-nodes
      // (#text is created by fast-xml-parser for mixed content — not a real XML tag)
      if (tagName.startsWith('?xml') || tagName === '#text') continue;

      const element = item[tagName];
      const nodeAttrs = item[':@'] || {};

      // Update context for this node — passed to children if we descend
      const updatedContext = this.updateContext(tagName, nodeAttrs, context);

      // Token gate: measure the full subtree content as embeddingText
      // Use parent context (not updatedContext) — the chunk's own tag is already in content
      const range = this.findElementRange(tagName, lines);
      const content = this.extractContent(lines, range);
      const embeddingText = this.createEmbeddingText(content, context);
      const tokenCount = this.countTokens(embeddingText);
      const lineSpan = range.end - range.start + 1;

      if (tokenCount <= this.maxTokens) {
        // --- This element FITS within the token limit ---
        //
        // ALL fit siblings are buffered together regardless of individual size.
        // A large-but-fitting sibling (e.g. a 5-line call-template) can absorb
        // preceding small siblings to jointly reach the MIN_CHUNK_LINES threshold.
        // Only oversized elements (exceed maxTokens) break the aggregation run.

        const isIsolated = this.isIsolatedFragment(element, nodeAttrs);

        // Compute combined token count if we add this element to the buffer
        const candidateContents = buffer ? [...buffer.contents, content] : [content];
        const candidateEmbedding = this.createAggregatedEmbeddingText(candidateContents, contextPrefix);
        const candidateTokens = this.countTokens(candidateEmbedding);
        const candidateLines = (buffer ? buffer.lineCount : 0) + lineSpan;

        if (buffer && candidateTokens > this.maxTokens) {
          // Adding this element would exceed the token limit.
          // Flush the existing buffer WITHOUT this element, then start a fresh buffer.
          const merged = this.flushAggregationBuffer(buffer, filePath, chunks, context, lastSiblingChunk);
          if (merged) lastSiblingChunk = merged;
          buffer = null;
        }

        const elementRefs = this.extractReferencesFromContent(content);

        if (buffer === null) {
          // Start a new buffer with this element
          const soloTokens = this.countTokens(
            this.createAggregatedEmbeddingText([content], contextPrefix)
          );
          buffer = {
            contents: [content],
            startLine: range.start,
            endLine: range.end,
            lineCount: lineSpan,
            tokenCount: soloTokens,
            references: elementRefs,
            contextPrefix,
            isAllIsolatedFragments: isIsolated,
          };
        } else {
          // Append to the existing buffer
          buffer.contents.push(content);
          buffer.endLine = range.end;
          buffer.lineCount = candidateLines;
          buffer.tokenCount = candidateTokens;
          buffer.references = [...buffer.references, ...elementRefs];
          buffer.isAllIsolatedFragments = buffer.isAllIsolatedFragments && isIsolated;
        }

        // Flush once the combined line count reaches the minimum threshold
        if (buffer.lineCount >= this.MIN_CHUNK_LINES) {
          const merged = this.flushAggregationBuffer(buffer, filePath, chunks, context, lastSiblingChunk);
          if (merged) lastSiblingChunk = merged;
          buffer = null;
        }

      } else if (Array.isArray(element)) {
        // --- OVERSIZED: exceeds token limit → descend into children ---
        // Flush any pending buffer first (oversized element always breaks the sibling run)
        if (buffer) {
          this.flushAggregationBuffer(buffer, filePath, chunks, context, lastSiblingChunk);
          buffer = null;
        }

        const childChunksBefore = chunks.length;
        this.processNode(element, lines, filePath, chunks, updatedContext);

        // Oversized leaf fallback: if no children produced chunks, force-emit this node
        if (chunks.length === childChunksBefore) {
          this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, context);
          lastSiblingChunk = chunks[chunks.length - 1];
        } else {
          lastSiblingChunk = null; // Invalidate since we emitted chunks at a deeper level
        }
      } else {
        // --- Leaf node (no children) that exceeds token limit → force-emit ---
        if (buffer) {
          this.flushAggregationBuffer(buffer, filePath, chunks, context, lastSiblingChunk);
          buffer = null;
        }
        this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, context);
        lastSiblingChunk = chunks[chunks.length - 1];
      }

      // Fix: Advance lastSearchPosition to prevent the next sibling search from
      // accidentally matching child tags inside the element we just skipped/processed.
      // range.end - 1 sets the search cursor at the closing tag's line, safely 
      // skipping internal lines.
      this.lastSearchPosition = Math.max(this.lastSearchPosition, range.end - 1);
    }

    // Flush any remaining buffered elements after all siblings are processed
    if (buffer) {
      this.flushAggregationBuffer(buffer, filePath, chunks, context, lastSiblingChunk);
    }
  }

  /**
   * Flush the aggregation buffer:
   * - If buffer is purely isolated fragments and `lastSiblingChunk` is provided,
   *   attempts backward-merging into it to prevent isolated standalone chunks.
   * - Otherwise, emits single element as a normal chunk, or multiple elements
   *   as a merged aggregated chunk.
   */
  private flushAggregationBuffer(
    buffer: AggregationBuffer,
    filePath: string,
    chunks: XMLChunk[],
    context: SemanticContext,
    lastSiblingChunk?: XMLChunk | null
  ): XMLChunk | null {
    // 1. BACKWARD MERGE INTERCEPTION
    // Intercept purely isolated fragments about to be emitted standalone
    if (buffer.isAllIsolatedFragments && lastSiblingChunk) {
      // Calculate embedding text additions without redundant context prefix
      const additionalEmbedding = this.createAggregatedEmbeddingText(buffer.contents, '').trim();
      const testEmbedding = lastSiblingChunk.embeddingText + (additionalEmbedding ? ' ' + additionalEmbedding : '');

      if (this.countTokens(testEmbedding) <= this.maxTokens) {
        // Merge! The preceding sibling absorbs this isolated fragment chunk
        lastSiblingChunk.content += '\n' + buffer.contents.join('\n');
        lastSiblingChunk.endLine = Math.max(lastSiblingChunk.endLine, buffer.endLine);
        lastSiblingChunk.embeddingText = testEmbedding;

        const combinedRefs = [...new Set([...(lastSiblingChunk.referencedSequences || []), ...buffer.references])];
        if (combinedRefs.length > 0) {
          lastSiblingChunk.referencedSequences = combinedRefs;
          if (lastSiblingChunk.context) {
            lastSiblingChunk.context.references = combinedRefs;
          }
        }
        return lastSiblingChunk;
      }
    }

    // 2. NORMAL FLUSH (Fallback or Forward Aggregation)
    const content = buffer.contents.length === 1 ? buffer.contents[0] : buffer.contents.join('\n');
    const embeddingText = this.createAggregatedEmbeddingText(buffer.contents, buffer.contextPrefix);
    const chunkIndex = this.chunkCounter++;
    const refs = [...new Set(buffer.references)];

    const newChunk: XMLChunk = {
      filePath,
      chunkType: 'aggregated',
      chunkIndex,
      startLine: buffer.startLine,
      endLine: buffer.endLine,
      content,
      embeddingText,
      context: { ...context, references: refs.length > 0 ? refs : undefined },
      sequenceKey: undefined,
      isSequenceDefinition: false,
      referencedSequences: refs,
    };

    chunks.push(newChunk);
    return newChunk;
  }

  /**
   * Create embedding text for an aggregated chunk.
   *
   * Context is prepended only ONCE (shared by all siblings), then each element's
   * cleaned content tokens are appended in order. This avoids context repetition
   * that would bloat the embedding and degrade retrieval quality.
   *
   * @param contents  Raw XML strings of each buffered element
   * @param contextPrefix  Pre-computed formatMetadata(context) string
   */
  private createAggregatedEmbeddingText(contents: string[], contextPrefix: string): string {
    const tokens: string[] = contextPrefix ? [contextPrefix] : [];

    for (const content of contents) {
      // Re-use the same XML cleaning pipeline as createEmbeddingText,
      // but without the context prefix (already added once above).
      const jsonBlocks: string[] = [];
      const jsonProtectedContent = content.replace(
        /<(format|args)[^>]*>([\s\S]*?)<\/\1>/g,
        (match, tag, jsonContent) => {
          const trimmed = jsonContent.trim();
          if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
            const placeholder = `__JSON_BLOCK_${jsonBlocks.length}__`;
            jsonBlocks.push(`${tag} ${trimmed}`);
            return placeholder;
          }
          return match;
        }
      );

      const cleanedContent = jsonProtectedContent
        .replace(/<([^>\/\s]+)([^>]*)>/g, ' $1 $2 ')
        .replace(/<\/[^>]+>/g, ' ')
        .replace(/<([^>\/\s]+)([^>]*)\s*\/>/g, ' $1 $2 ')
        .replace(/="([^"]*)"/g, '=$1')
        .replace(/='([^']*)'/g, '=$1')
        .replace(/__JSON_BLOCK_(\d+)__/g, (_, idx) => ` ${jsonBlocks[parseInt(idx)]} `)
        .replace(/[^\w\s=\$\{\}\[\]\/\-\.,:@]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

      const contentTokens = cleanedContent
        .split(/\s+/)
        .filter(t => (t.length > 1 || /^\d+$/.test(t)) && t.length < 100);

      tokens.push(...contentTokens);
    }

    return tokens.join(' ');
  }

  /**
   * Update semantic context as we traverse the tree.
   * FULLY GENERIC: reads directly from the parsed tree — no registry.
   */
  private updateContext(tagName: string, attrs: Record<string, string>, parentContext: SemanticContext): SemanticContext {
    const newContext = { ...parentContext };
    const localName = tagName.split(':').pop() || tagName;

    // Skip the root artifact tag — context.artifact was already set by buildRootContext.
    // Re-adding it here would duplicate it as a dynamic context key.
    if (tagName === parentContext.artifact?.type || localName === parentContext.artifact?.type) {
      return newContext;
    }

    // Generic context: capture all attributes for any element encountered during traversal.
    // Any attribute could be semantically important (e.g., methods, uri-template, xpath).
    const allAttrs = this.extractAllAttributes(attrs);

    if (Object.keys(allAttrs).length > 0) {
      newContext[localName] = allAttrs;
    } else {
      // No attributes (e.g., <then>, <else>, <inSequence>) — store as a string marker
      newContext[localName] = localName;
    }

    return newContext;
  }

  /**
   * Extract ALL non-internal attributes from an element, cleaning prefixes.
   * Used for artifact-level elements where every attribute is configuration-critical.
   */
  private extractAllAttributes(attrs: Record<string, string>): Record<string, any> {
    const allAttrs: Record<string, any> = {};
    for (const [key, value] of Object.entries(attrs)) {
      if (!key.startsWith(':@') && !key.startsWith('@_')) {
        allAttrs[key] = value;
      } else if (key.startsWith('@_')) {
        allAttrs[key.substring(2)] = value;
      }
    }
    return allAttrs;
  }

  /**
   * Create a chunk from the current node
   */
  private createChunk(
    tagName: string,
    attrs: Record<string, string>,
    content: string,
    range: LineRange,
    filePath: string,
    chunks: XMLChunk[],
    context: SemanticContext
  ): void {
    const chunkIndex = this.chunkCounter++;

    const embeddingText = this.createEmbeddingText(content, context);

    // Extract references from this chunk's content.
    // NOTE: We do NOT mutate the shared `context` object here — that would
    // pollute the context passed to any sibling nodes processed afterwards.
    const chunkReferences = this.extractReferencesFromContent(content);

    // A chunk is a standalone artifact definition when its tag IS the root artifact tag.
    // This is true exactly when this chunk represents the top-level element of the file.
    const isDefinition = tagName === context.artifact?.type;
    const sequenceKey = isDefinition
      ? (attrs.name || attrs['@_name'] || attrs.key || attrs['@_key'])
      : undefined;

    chunks.push({
      filePath,
      chunkType: tagName,
      chunkIndex,
      startLine: range.start,
      endLine: range.end,
      content,
      embeddingText,
      context: { ...context, references: chunkReferences.length > 0 ? chunkReferences : undefined },
      sequenceKey,
      isSequenceDefinition: isDefinition,
      referencedSequences: chunkReferences,
    });
  }



  /**
   * Count tokens using the embedding model's tokenizer.
   *
   * Always uses the same AutoTokenizer that the embedding model uses, so the
   * token gate is exact — no char/4 approximation that can undercount.
   * `initialize()` must have been called before this (guaranteed by `chunkFile`).
   */
  private countTokens(content: string): number {
    if (this.tokenizer) {
      return this.tokenizer.encode(content).length;
    }

    // Should never reach here after initialize() — but throw loudly if it does
    // so the problem is visible rather than silently producing wrong counts.
    throw new Error('Tokenizer not initialized. Call initialize() before counting tokens.');
  }

  /**
   * Format context metadata into text for token counting and embedding.
   * FULLY GENERIC: Iterates all context keys uniformly.
   * No hardcoded field-specific formatting.
   */
  private formatMetadata(context: SemanticContext): string {
    const parts: string[] = [];

    // 1. Artifact context (root-level metadata)
    if (context.artifact) {
      const { type, name, xmlns, ...rest } = context.artifact;
      parts.push(`${this.formatContextKey(type)}: ${name}`);
      // Include additional artifact attrs (context, transports, etc.)
      const extraPairs = Object.entries(rest)
        .filter(([k, v]) => v !== undefined && v !== null && v !== '' && k !== 'isCustom' && k !== 'rootTag' && k !== 'inferredFromPath')
        .map(([k, v]) => `${k}=${v}`)
        .join(' ');
      if (extraPairs) parts.push(extraPairs);
    }

    // 2. DYNAMIC CONTEXT: Format ALL other context fields uniformly
    //    This handles resource, sequence, filter, query, operation, and ANY arbitrary element
    const skipKeys = new Set(['artifact', 'references']);

    for (const [key, value] of Object.entries(context)) {
      if (skipKeys.has(key) || value === undefined || value === null) continue;

      const formattedKey = this.formatContextKey(key);

      if (typeof value === 'string') {
        // Simple string context (e.g., sequence name)
        parts.push(`${formattedKey}: ${value}`);
      } else if (typeof value === 'object' && !Array.isArray(value)) {
        // Object context with attributes
        const attrPairs = Object.entries(value)
          .filter(([k, v]) => v !== undefined && v !== null && v !== '')
          .map(([k, v]) => `${k}=${v}`)
          .join(' ');
        if (attrPairs) {
          parts.push(`${formattedKey}: ${attrPairs}`);
        }
      }
    }

    // 3. References (if any)
    if (context.references && context.references.length > 0) {
      parts.push(`Uses: ${context.references.join(', ')}`);
    }

    return parts.join(' ');
  }

  /**
   * Format context key for display (e.g., "filter" -> "Filter", "Policy" -> "Policy")
   */
  private formatContextKey(key: string): string {
    return key.charAt(0).toUpperCase() + key.slice(1);
  }

  /**
   * Find the line range for an XML element.
   * Automatically includes structural wrapper elements (onAccept, onReject, then, else, etc.)
   */
  private findElementRange(tagName: string, lines: string[]): LineRange {
    let startLine = -1;
    let endLine = -1;
    let depth = 0;

    for (let i = this.lastSearchPosition; i < lines.length; i++) {
      const line = lines[i];

      if (startLine === -1) {
        const openPattern = new RegExp(`<${tagName}[\\s>/]`);
        if (openPattern.test(line)) {
          startLine = i + 1;
          this.lastSearchPosition = i + 1;

          if (line.includes('/>')) {
            endLine = i + 1;
            break;
          }
          depth = 1;
        }
      } else {
        const openPattern = new RegExp(`<${tagName}[\\s>]`);
        const closePattern = new RegExp(`</${tagName}>`);

        if (openPattern.test(line) && !line.includes('/>')) {
          depth++;
        }
        if (closePattern.test(line)) {
          depth--;
          if (depth === 0) {
            endLine = i + 1;
            break;
          }
        }
      }
    }

    if (startLine === -1) startLine = 1;
    if (endLine === -1) endLine = startLine;

    return { start: startLine, end: endLine };
  }


  private extractContent(lines: string[], range: LineRange): string {
    return lines.slice(range.start - 1, range.end).join('\n');
  }



  /**
   * Create natural text representation for embedding.
   * Format: [Formatted Context Metadata] + [Cleaned XML Content tokens]
   *
   * Example:
   *   context → "Api: BankAPI context=/bankapi Resource: method=GET uriTemplate=/"
   *   content → <payloadFactory><format>{"greeting":"Hello"}</format></payloadFactory>
   *   → "Api: BankAPI context=/bankapi Resource: method=GET payloadFactory format greeting Hello"
   */
  private createEmbeddingText(
    content: string,
    context: SemanticContext
  ): string {

    // Start with formatted context metadata as text
    const contextStr = this.formatMetadata(context);
    const tokens: string[] = contextStr ? [contextStr] : [];

    // JSON BLOCK PROTECTION: Preserve JSON inside format/args tags before cleaning
    // This prevents breaking structured payloads in embedding text
    const jsonBlocks: string[] = [];
    const jsonProtectedContent = content.replace(
      /<(format|args)[^>]*>([\s\S]*?)<\/\1>/g,
      (match, tag, jsonContent) => {
        // Check if the content looks like JSON
        const trimmed = jsonContent.trim();
        if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
          const placeholder = `__JSON_BLOCK_${jsonBlocks.length}__`;
          jsonBlocks.push(`${tag} ${trimmed}`);
          return placeholder;
        }
        return match;
      }
    );

    // Comprehensive XML preprocessing: Remove all angle brackets and create natural text
    const cleanedContent = jsonProtectedContent
      // Extract tag names and attributes from opening tags: <tag attr="val"> → tag attr="val"
      .replace(/<([^>\/\s]+)([^>]*)>/g, ' $1 $2 ')
      // Remove closing tags: </tag> → (empty)
      .replace(/<\/[^>]+>/g, ' ')
      // Extract from self-closing tags: <tag attr="val"/> → tag attr="val"
      .replace(/<([^>\/\s]+)([^>]*)\s*\/>/g, ' $1 $2 ')
      // Clean up attribute formatting: attr="value" → attr=value
      .replace(/="([^"]*)"/g, '=$1')
      .replace(/='([^']*)'/g, '=$1')
      // Restore JSON blocks
      .replace(/__JSON_BLOCK_(\d+)__/g, (_, idx) => ` ${jsonBlocks[parseInt(idx)]} `)
      // Remove remaining special characters but preserve $, {, }, [, ] for expressions and paths
      .replace(/[^\w\s=\$\{\}\[\]\/\-\.,:@]/g, ' ')
      // Normalize whitespace
      .replace(/\s+/g, ' ')
      .trim();

    // Split into meaningful tokens
    const contentTokens = cleanedContent
      .split(/\s+/)
      .filter(t => (t.length > 1 || /^\d+$/.test(t)) && t.length < 100); // Preserve numeric values (e.g. 0, 1) and longer tokens

    tokens.push(...contentTokens);

    return tokens.join(' ');
  }
}
