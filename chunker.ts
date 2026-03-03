import * as fs from 'fs';
import { createHash } from 'crypto';
import { XMLParser } from 'fast-xml-parser';
import { AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';
import { ArtifactRegistry, artifactRegistry, ArtifactMetadata } from './artifact-registry';
import { config } from './config';

/**
 * Semantic, Hierarchical, Structure-Aware XML Chunker for WSO2 MI artifacts
 *
 * Uses plugin-based ArtifactRegistry for extensible artifact detection.
 */

export interface XMLChunk {
  filePath: string;
  chunkType: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  content: string;
  embeddingText: string;
  contentHash: string;
  context: SemanticContext;
  sequenceKey?: string;
  isSequenceDefinition?: boolean;
  referencedSequences?: string[];
}

/**
 * Semantic context — fully generic, schema-agnostic.
 *
 * DESIGN: Only two explicit fields exist:
 *   - `artifact`: Root-level artifact metadata (detected via registry or heuristics)
 *   - `references`: Cross-artifact references extracted from content
 *
 * All other context (resource boundaries, sequence names, filters, etc.) is stored
 * dynamically via the `[key: string]: any` index signature. This means the chunker
 * works identically for `<api>/<resource>/<inSequence>` and `<aaappp>/<reesss>/<insq>`.
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

/**
 * Compute a deterministic content hash for a chunk.
 * Hashes raw XML content + context.
 * Stored as `content_hash` in the DB — drives incremental update logic:
 * unchanged hash → reuse embedding, changed hash → re-embed.
 */
function computeChunkHash(
  xmlContent: string,
  context: Record<string, any>
): string {
  const hashInput = JSON.stringify({ xml: xmlContent, context });
  return createHash('sha256').update(hashInput).digest('hex');
}

interface LineRange {
  start: number;
  end: number;
}

export class XMLChunker {
  private chunkCounter = 0;
  private lastSearchPosition: number = 0;
  private readonly maxTokens: number;
  private embedder: any;
  private registry: ArtifactRegistry;
  private tokenizer: PreTrainedTokenizer | null = null;

  constructor(embedder?: any, registry?: ArtifactRegistry) {
    this.embedder = embedder;
    this.registry = registry || artifactRegistry;
    this.maxTokens = config.maxTokens; // Default: 256
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

    // Detect artifact type using registry
    const rootContext = this.buildRootContext(parsed, filePath);

    this.processNode(parsed, lines, filePath, chunks, rootContext);

    return chunks;
  }

  /**
   * Build root context by detecting artifact type from XML.
   * Uses registry instead of path-based detection.
   */
  private buildRootContext(parsed: any, filePath: string): SemanticContext {
    const context: SemanticContext = {};

    // Try to detect artifact type from XML structure
    const detected = this.registry.detectArtifactType(parsed);

    if (detected) {
      const { metadata } = detected;

      // UNIFORM: All artifact types stored in context.artifact
      // No special-casing for api/proxy/sequence — fully generic
      context.artifact = {
        type: metadata.type,
        name: metadata.name,
        xmlns: metadata.xmlns,
        ...metadata.additionalInfo,
      };
    } else {
      // Fallback: detect any artifact (including custom/unregistered types)
      // Pass filePath to infer type from folder structure
      const anyArtifact = this.registry.detectAnyArtifact(parsed, filePath);
      if (anyArtifact) {
        context.artifact = {
          type: anyArtifact.type,
          name: anyArtifact.name,
          xmlns: anyArtifact.xmlns,
          ...anyArtifact.additionalInfo,
        };
      } else {
        // Ultimate fallback if parsing completely fails
        context.artifact = {
          type: 'unknown',
          name: 'unknown',
        };
      }
    }

    return context;
  }

  /**
   * SEMANTIC BOUNDARY DETECTION (Registry-Based)
   *
   * Queries the artifact registry instead of hardcoded lists.
   * Falls back to heuristics for unknown tags.
   */
  private isSemanticBoundary(tagName: string, attrs: Record<string, string> = {}, element?: any, parentTagName?: string): boolean {
    const localName = tagName.split(':').pop() || tagName;

    // 1. Registry Lookup (Explicit)
    // Check both full name (wsp:Policy) and local name (Policy)
    if (this.registry.isSemanticBoundary(tagName) || this.registry.isSemanticBoundary(localName)) {
      return true;
    }

    // 2. Dot Notation Rule (Mediators & Connectors)
    // e.g., http.post, google.spreadsheet, ENTC.agent
    // The dot is part of the tag name, NOT a namespace separator in this context
    if (tagName.includes('.')) {
      return true;
    }

    // 3. Namespace Pattern Rule (WS-* & Extensions)
    // Matches: lowercase_prefix:CamelCaseLocalName
    // e.g., wsp:Policy, throttle:ThrottleAssertion
    if (tagName.includes(':')) {
      const [prefix, localNamePart] = tagName.split(':', 2);
      // Heuristic: Prefix is lowercase alpha, LocalName starts with Uppercase
      const isNamespacePattern = /^[a-z]+$/.test(prefix) && /^[A-Z]/.test(localNamePart);
      if (isNamespacePattern) {
        return true;
      }
    }

    // 4. CamelCase Config Rule (Declarative Configuration)
    // e.g., Filter, ThrottleAssertion, MaximumConcurrentAccess
    // Legacy support for tags that behave like classes/objects
    // Exclude simple lowercase tags unless they match specific keywords
    if (/^[A-Z]/.test(localName) && !localName.includes('.')) {
      return true;
    }

    // 5. Standard Data Service / Flow Keywords
    const standardKeywords = ['query', 'operation', 'resource', 'config', 'validate', 'header'];
    if (standardKeywords.includes(localName)) {
      return true;
    }

    // 6. Universal Fallback (The Safety Net)
    // Rule A: Has identifying attributes (name, key, id, etc.)
    const attrCount = Object.keys(attrs).filter(k => !k.startsWith('#')).length;
    if (attrCount > 0) {
      return true;
    }

    // Rule B: Structural Complexity (Has multiple distinct children)
    // If it contains logic/structure, it's likely a container we want to chunk
    if (element && this.hasComplexStructure(element)) {
      return true;
    }

    // 7. Connector Child Rule (General)
    // If the immediate parent is a connector/mediator with a dot in its tag name
    // (e.g., <ai.agent>, <http.post>, <email.send>), ALL direct children are
    // configuration properties of that connector and should be chunked together.
    // This is purely structural — no hardcoded tag names anywhere.
    if (parentTagName && parentTagName.includes('.')) {
      return true;
    }

    return false;
  }

  /**
   * Check if element has complex nested structure (multiple distinct child tags)
   */
  private hasComplexStructure(element: any): boolean {
    if (!element || typeof element !== 'object') return false;

    // Count distinct child tags (exclude attributes, text, processing instructions)
    const childTags = Object.keys(element).filter(key =>
      !key.startsWith(':@') &&
      !key.startsWith('#') &&
      !key.startsWith('?')
    );

    return childTags.length >= 2;
  }

  /**
   * Check if tag is a resource type (uses registry)
   */
  private isResourceType(tagName: string): boolean {
    const localName = tagName.split(':').pop() || tagName;
    return this.registry.isResourceType(tagName) || this.registry.isResourceType(localName);
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
   * EXCLUSIVE TOP-DOWN CHUNKING with token gating
   *
   * Fully structure-based: chunkability is determined by registry + heuristics,
   * never by hardcoded tag names. Includes oversized leaf fallback.
   */
  private processNode(
    node: any,
    lines: string[],
    filePath: string,
    chunks: XMLChunk[],
    context: SemanticContext,
    parentTagName?: string  // Tag name of the element that triggered this descent
  ): void {
    if (!Array.isArray(node)) return;

    for (const item of node) {
      const tagName = Object.keys(item).find(key => key !== ':@') || '';
      if (!tagName) continue;

      // Skip XML declaration and other processing instructions
      if (tagName.startsWith('?xml')) continue;

      const element = item[tagName];
      const nodeAttrs = item[':@'] || {};

      // Update context for THIS node (will be passed to children)
      const updatedContext = this.updateContext(tagName, nodeAttrs, context);

      // Check if this is a chunkable node
      // Pass parentTagName so Rule 7 (Connector Child) can fire for scalar children of connectors
      const isChunkable = this.isResourceType(tagName) ||
        this.isSemanticBoundary(tagName, nodeAttrs, element, parentTagName);

      if (isChunkable) {
        // Token gating: Check if subtree fits within limit.
        // Gate on the actual embeddingText (cleaned XML + context) — the same text that will be
        // embedded — so the token limit is checked consistently. Raw XML over-counts because
        // tag brackets, closing tags, and quote characters are stripped during cleaning.
        const range = this.findElementRange(tagName, lines);
        const content = this.extractContent(lines, range);
        // CRITICAL: Use parent context (not updatedContext) to avoid duplication
        // The chunk's own attributes are already in the content, so we should NOT include them in metadata
        const embeddingText = this.createEmbeddingText(tagName, content, nodeAttrs, context);
        const tokenCount = this.countTokens(embeddingText);

        if (tokenCount <= this.maxTokens) {
          // Subtree fits → Emit chunk with parent context (not updatedContext)
          this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, context);
        } else {
          // Subtree too large → Descend to children with updated context, passing THIS tag as parent
          if (Array.isArray(element)) {
            const childChunksBefore = chunks.length;
            this.processNode(element, lines, filePath, chunks, updatedContext, tagName);

            // OVERSIZED LEAF FALLBACK: If no children produced any chunks,
            // this is a leaf-like node that exceeds maxTokens.
            // Force-emit it as a chunk rather than silently dropping content.
            if (chunks.length === childChunksBefore) {
              this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, context);
            }
          } else {
            // Atomic node with no children that exceeds maxTokens → force-emit
            const range = this.findElementRange(tagName, lines);
            const content = this.extractContent(lines, range);
            this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, context);
          }
        }
      } else if (Array.isArray(element)) {
        // Non-chunkable container → traverse children, passing THIS tag as the parent
        this.processNode(element, lines, filePath, chunks, updatedContext, tagName);
      } else if (typeof element === 'string' && element.trim().length > 0 && parentTagName && parentTagName.includes('.')) {
        // LEAF TEXT NODE inside a connector (e.g., <role> inside <ai.agent>):
        // The parent connector made this node chunkable via Rule 7, but fast-xml-parser
        // returns the text content as a raw string, not an array — so the normal
        // isChunkable path never fires for it. Handle it explicitly here.
        // We find its range and emit a chunk so no config property is ever silently dropped.
        const range = this.findElementRange(tagName, lines);
        const content = this.extractContent(lines, range);
        if (content.trim().length > 0) {
          this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, context);
        }
      }
    }
  }

  /**
   * Update semantic context as we traverse the tree.
   * FULLY GENERIC: No hardcoded tag names. Uses registry for artifact roots,
   * attribute-based heuristics for all other elements.
   */
  private updateContext(tagName: string, attrs: Record<string, string>, parentContext: SemanticContext): SemanticContext {
    const newContext = { ...parentContext };
    const localName = tagName.split(':').pop() || tagName;

    // 1. Check if this is a REGISTERED ARTIFACT ROOT TAG (via registry)
    //    e.g., api, proxy, sequence, endpoint, inboundEndpoint, data, etc.
    const plugin = this.registry.getPluginForRootTag(tagName) || this.registry.getPluginForRootTag(localName);
    if (plugin) {
      // Extract metadata using the plugin's own extractor
      const metadata = plugin.extractMetadata(tagName, attrs);
      const allAttrs = this.extractAllAttributes(attrs);
      newContext.artifact = {
        type: metadata.type,
        name: metadata.name,
        xmlns: metadata.xmlns,
        ...metadata.additionalInfo,
        ...allAttrs,
      };
    } else {
      // 2. GENERIC CONTEXT: For ALL other elements
      //    Capture ALL attributes (not just a whitelist) — any attribute could be
      //    semantically important in arbitrary XML (e.g., methods, uri-template, href)
      const allAttrs = this.extractAllAttributes(attrs);

      if (Object.keys(allAttrs).length > 0) {
        // Has attributes → store as object (e.g., resource: { methods: 'POST', 'uri-template': '/deposit' })
        newContext[localName] = allAttrs;
      } else {
        // No attributes (e.g., <then>, <else>, <onAccept>, <inSequence>)
        // Always add as string context — every element in the traversal path is meaningful.
        // updateContext is only called for element nodes, never for text/leaf content.
        newContext[localName] = localName;
      }
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

    const embeddingText = this.createEmbeddingText(tagName, content, attrs, context);
    const contentHash = computeChunkHash(content, context);

    // Extract references from this chunk's content
    const chunkReferences = this.extractReferencesFromContent(content);
    if (chunkReferences.length > 0) {
      context.references = chunkReferences;
    }

    // Detect if this is a standalone artifact definition
    const standaloneTypes = ['sequence', 'localEntry', 'endpoint', 'template'];
    const isStandalone = standaloneTypes.includes(tagName);
    const sequenceKey = isStandalone ? (attrs.name || attrs['@_name'] || attrs.key || attrs['@_key']) : undefined;

    chunks.push({
      filePath,
      chunkType: tagName,
      chunkIndex,
      startLine: range.start,
      endLine: range.end,
      content,
      embeddingText,
      contentHash,
      context: { ...context, references: chunkReferences.length > 0 ? chunkReferences : undefined },
      sequenceKey,
      isSequenceDefinition: isStandalone,
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
  private countTokens(content: string, metadata: string = ''): number {
    const fullText = (metadata ? metadata + ' ' : '') + content;

    if (this.tokenizer) {
      return this.tokenizer.encode(fullText).length;
    }

    // Should never reach here after initialize() — but throw loudly if it does
    // so the problem is visible rather than silently producing wrong counts.
    throw new Error('Tokenizer not initialized. Call initialize() before counting tokens.');
  }

  /**
   * Extract node name from element attributes.
   */
  private getNodeName(tagName: string, element: any): string {
    const attrs = this.extractAttributes(element);
    return attrs.name || attrs['@_name'] || attrs.key || attrs['@_key'] ||
      attrs.context || attrs['@_context'] || tagName;
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

  private extractAttributes(element: any): Record<string, string> {
    const attrs: Record<string, string> = {};

    if (Array.isArray(element)) {
      for (const item of element) {
        if (item[':@']) {
          Object.assign(attrs, item[':@']);
          break;
        }
      }
    } else if (element && element[':@']) {
      Object.assign(attrs, element[':@']);
    }

    return attrs;
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

    // GENERALIZABLE WRAPPER DETECTION:
    // Expand range to include structural wrapper elements (onAccept, onReject, then, else, etc.)
    // These are parent elements that have minimal/no attributes and provide structural context
    const expandedRange = this.expandRangeForStructuralWrappers(startLine, endLine, lines);

    return expandedRange;
  }

  /**
   * Expand a range to include structural wrapper elements.
   * Detects wrappers generically without hardcoding tag names.
   */
  private expandRangeForStructuralWrappers(startLine: number, endLine: number, lines: string[]): LineRange {
    let newStart = startLine;
    let newEnd = endLine;

    // Look backwards for structural wrapper opening tags
    // A structural wrapper is typically:
    // - A simple opening tag with no or minimal attributes
    // - Located immediately before our element (with possible whitespace)
    if (startLine > 1) {
      for (let i = startLine - 2; i >= 0 && i >= startLine - 5; i--) {
        const line = lines[i].trim();

        // Check if this line is a simple opening tag (e.g., <onAccept>, <then>, <else>)
        // Pattern: <tagname> or <tagname >, but NOT tags with attributes like <tag attr="value">
        const simpleOpeningTag = /^<(\w+:?\w*)>\s*$/;
        const match = line.match(simpleOpeningTag);

        if (match) {
          // Found a structural wrapper, expand to include it
          newStart = i + 1;
          // Continue looking for more nested wrappers
        } else if (line && !line.startsWith('<!--') && line !== '') {
          // Hit a non-wrapper line, stop searching
          break;
        }
      }
    }

    // Look forwards for corresponding closing tags
    // Match each wrapper we found when expanding backwards
    if (newStart < startLine && endLine < lines.length) {
      const wrappersToClose = startLine - newStart;
      let closedWrappers = 0;

      for (let i = endLine; i < lines.length && i < endLine + 10; i++) {
        const line = lines[i].trim();

        // Check if this is a simple closing tag
        const simpleClosingTag = /^<\/(\w+:?\w*)>\s*$/;
        if (simpleClosingTag.test(line)) {
          closedWrappers++;
          newEnd = i + 1;

          if (closedWrappers >= wrappersToClose) {
            break;
          }
        } else if (line && !line.startsWith('<!--') && line !== '') {
          // Hit a non-wrapper line before closing all wrappers
          break;
        }
      }
    }

    return { start: newStart, end: newEnd };
  }

  private extractContent(lines: string[], range: LineRange): string {
    return lines.slice(range.start - 1, range.end).join('\n');
  }



  /**
   * Create natural text representation for embedding.
   * Format: [JSON Context] + [Cleaned XML Content]
   *
   * Example:
   *   Context: {"api":{"name":"BankAPI","context":"/bankapi"},"resource":{"method":"GET","uriTemplate":"/"}}
   *   Content: <payloadFactory><format>{"greeting":"Hello"}</format></payloadFactory>
   *   → {"api":{"name":"BankAPI","context":"/bankapi"},"resource":{"method":"GET"}} payloadFactory format greeting Hello
   */
  private createEmbeddingText(
    tagName: string,
    content: string,
    attrs: Record<string, string>,
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
