import * as fs from 'fs';
import { XMLParser } from 'fast-xml-parser';
import { AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';
import { computeChunkHash } from './content-hash';
import { config } from './config';

/**
 * Semantic Structure-Aware XML Chunker (Generalized)
 * 
 * Works with ANY XML configuration file without hardcoded artifact knowledge.
 * Uses XML structure heuristics for semantic boundary detection, context tracking,
 * and hierarchical chunking. No plugin registry required.
 * 
 * Chunking strategy:
 *   1. Detect root artifact type generically from the XML root element
 *   2. Traverse the XML tree top-down
 *   3. At each node, determine if it's a semantic boundary using structural heuristics:
 *      - Has identifying attributes (name, key, id, etc.)
 *      - Is a root-level resource element
 *      - Has dotted name (connector mediator pattern)
 *   4. If the node's subtree fits within the token limit -> emit as a chunk
 *   5. If too large -> descend into children
 *   6. Track hierarchical context (parent element attributes) to enrich embedding text
 * 
 * Tokenization via sentence-transformers/all-MiniLM-L6-v2 AutoTokenizer.
 */

export interface XMLChunk {
  filePath: string;
  resourceName: string;
  resourceType: string;
  chunkType: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  content: string;
  parentChunkId: number | null;
  embeddingText: string;
  semanticType: string;
  semanticIntent: string;
  contentHash: string;
  context: SemanticContext;
  sequenceKey?: string;
  isSequenceDefinition?: boolean;
  referencedSequences?: string[];
}

/**
 * Semantic context built generically from XML structure.
 * Tracks ancestor element attributes to provide hierarchical context.
 */
export interface SemanticContext {
  // Populated when root element has both 'name' and 'context' attributes (API-like pattern)
  api?: {
    name?: string;
    context?: string;
    xmlns?: string;
  };
  // Populated for elements that have method + URI pattern attributes
  resource?: {
    method?: string;
    uriTemplate?: string;
  };
  // Populated for sequence-like context (either string tag name or {name, xmlns} object)
  sequence?: string | {
    name?: string;
    xmlns?: string;
  };
  // Generic artifact context for root elements that don't match api/sequence patterns
  artifact?: {
    type: string;
    name: string;
    xmlns?: string;
    [key: string]: any;
  };
  // Populated for elements with id + useConfig pattern (query-like)
  query?: {
    id?: string;
    useConfig?: string;
  };
  // Populated for elements with a 'name' attr under a data-style root (operation-like)
  operation?: {
    name?: string;
    callsQuery?: string;
  };
  references?: string[];
  // Allow dynamic extension
  [key: string]: any;
}

interface LineRange {
  start: number;
  end: number;
}

/**
 * Metadata extracted generically from the root element of an XML document.
 */
interface RootArtifactInfo {
  /** The XML root tag name (e.g., 'api', 'sequence', 'data', 'localEntry') */
  rootTag: string;
  /** The artifact type derived from the root tag or folder path */
  type: string;
  /** The artifact name derived from common attribute patterns */
  name: string;
  /** XML namespace if present */
  xmlns?: string;
  /** Additional attributes from the root element */
  additionalAttrs: Record<string, string>;
}

/**
 * Well-known XML attribute names that identify an element.
 */
const IDENTITY_ATTRIBUTES = ['name', 'key', 'id', 'context'];

/**
 * Tags whose names end with "Sequence" indicate a sequence context boundary.
 * This is a structural naming convention, not a hardcoded artifact list.
 */
function isSequenceLikeTag(tagName: string): boolean {
  return tagName.endsWith('Sequence');
}

/**
 * Extract a human-readable name from an element's attributes using common patterns.
 */
function extractName(attrs: Record<string, string>): string {
  for (const attr of IDENTITY_ATTRIBUTES) {
    const val = attrs[attr] || attrs[`@_${attr}`];
    if (val) return val;
  }
  return 'unknown';
}

export class XMLChunker {
  private nextChunkIndex = 0;
  private lineSearchCursor: number = 0;
  private readonly maxTokens: number;
  private tokenizer: PreTrainedTokenizer | null = null;
  private rootArtifact: RootArtifactInfo | null = null;

  constructor() {
    this.maxTokens = config.maxTokens;
  }

  /**
   * Initialize the tokenizer (must be called before chunking)
   * Loads sentence-transformers/all-MiniLM-L6-v2 AutoTokenizer
   */
  async initialize(): Promise<void> {
    if (!this.tokenizer) {
      this.tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
    }
  }

  async chunkFile(filePath: string): Promise<XMLChunk[]> {
    await this.initialize();
    this.nextChunkIndex = 0;
    this.lineSearchCursor = 0;
    this.rootArtifact = null;
    const xmlContent = await fs.promises.readFile(filePath, 'utf-8');
    const lines = xmlContent.split('\n');

    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '',
      removeNSPrefix: true,
      preserveOrder: true,
      alwaysCreateTextNode: false,
    });

    const parsed = parser.parse(xmlContent);
    const chunks: XMLChunk[] = [];

    // Detect root artifact type generically from XML structure
    this.rootArtifact = this.detectRootArtifact(parsed, filePath);
    const rootContext = this.buildRootContext(this.rootArtifact);

    this.processNode(parsed, lines, filePath, chunks, null, rootContext);

    return chunks;
  }

  /**
   * Detect root artifact info purely from XML structure.
   * Extracts the first non-processing-instruction root element,
   * reads its tag name and attributes to determine type and name.
   * Falls back to folder path inference for type when the tag name
   * doesn't directly indicate the artifact type.
   */
  private detectRootArtifact(parsed: any, filePath: string): RootArtifactInfo {
    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        const tagName = Object.keys(item).find(key => key !== ':@');
        if (!tagName || tagName === '?xml') continue;

        const attrs = item[':@'] || {};
        const name = extractName(attrs);
        const xmlns = attrs.xmlns || attrs['@_xmlns'];

        // Infer type from tag name and folder path
        const type = this.inferArtifactType(tagName, filePath);

        // Collect all non-internal attributes for additional info
        const additionalAttrs: Record<string, string> = {};
        for (const [k, v] of Object.entries(attrs)) {
          if (!k.startsWith('#') && k !== 'xmlns' && k !== '@_xmlns') {
            additionalAttrs[k.replace(/^@_/, '')] = String(v);
          }
        }

        return { rootTag: tagName, type, name, xmlns, additionalAttrs };
      }
    }

    return { rootTag: 'unknown', type: 'unknown', name: 'unknown', additionalAttrs: {} };
  }

  /**
   * Infer artifact type from root tag name and file path.
   * 
   * Strategy:
   *   1. Well-known structural tag-to-type mappings (e.g., 'data' -> 'dataService')
   *   2. Tag name used directly if it's self-descriptive
   *   3. Folder path patterns as fallback for custom/unknown tags
   *   4. Tag name itself as ultimate fallback
   */
  private inferArtifactType(tagName: string, filePath: string): string {
    // Structural tag-to-type mappings for ambiguous tag names
    const tagToType: Record<string, string> = {
      data: 'dataService',
      datasource: 'dataSource',
      proxy: 'proxyService',
    };

    if (tagToType[tagName]) return tagToType[tagName];

    // Self-descriptive tag names that can be used directly as types
    const selfDescriptiveTags = [
      'api', 'sequence', 'localEntry', 'endpoint', 'template',
      'task', 'messageStore', 'messageProcessor', 'inboundEndpoint',
    ];
    if (selfDescriptiveTags.includes(tagName)) return tagName;

    // Folder-path-based inference for custom/unknown root tags
    const folderPatterns: [RegExp, string][] = [
      [/\/data-sources\//,        'dataSource'],
      [/\/apis\//,                'api'],
      [/\/proxy-services\//,      'proxyService'],
      [/\/sequences\//,           'sequence'],
      [/\/endpoints\//,           'endpoint'],
      [/\/local-entries\//,       'localEntry'],
      [/\/templates\//,           'template'],
      [/\/data-services\//,       'dataService'],
      [/\/tasks\//,               'task'],
      [/\/message-stores\//,      'messageStore'],
      [/\/message-processors\//,  'messageProcessor'],
      [/\/inbound-endpoints\//,   'inboundEndpoint'],
    ];

    for (const [pattern, type] of folderPatterns) {
      if (pattern.test(filePath)) return type;
    }

    return tagName;
  }

  /**
   * Build the root semantic context from detected artifact info.
   * 
   * Uses structural patterns to decide how to populate context:
   *   - Root element with 'name' + 'context' attributes -> API-like (sets context.api)
   *   - Root type is 'sequence' -> Sequence (sets context.sequence as {name, xmlns})
   *   - Everything else -> generic artifact (sets context.artifact)
   */
  private buildRootContext(info: RootArtifactInfo): SemanticContext {
    const context: SemanticContext = {};

    // Pattern: root element has both 'name' and 'context' attributes -> API-like
    if (info.additionalAttrs.name && info.additionalAttrs.context) {
      context.api = {
        name: info.additionalAttrs.name,
        context: info.additionalAttrs.context,
        xmlns: info.xmlns,
      };
    }
    // Pattern: root tag represents a named sequence
    else if (info.type === 'sequence') {
      context.sequence = {
        name: info.name,
        xmlns: info.xmlns,
      };
    }
    // Everything else: generic artifact context
    else {
      context.artifact = {
        type: info.type,
        name: info.name,
        xmlns: info.xmlns,
        ...this.extractAdditionalInfo(info),
      };
    }

    return context;
  }

  /**
   * Extract additional info from root artifact attributes.
   * Adds extra context for structural patterns found in the root element.
   */
  private extractAdditionalInfo(info: RootArtifactInfo): Record<string, any> {
    const extra: Record<string, any> = {};

    if (info.additionalAttrs.transports) extra.transports = info.additionalAttrs.transports;
    if (info.additionalAttrs.class) extra.className = info.additionalAttrs.class;
    if (info.additionalAttrs.protocol) extra.protocol = info.additionalAttrs.protocol;
    if (info.additionalAttrs.group) extra.group = info.additionalAttrs.group;

    // For data service root elements, always include enableBatchRequests
    if (info.type === 'dataService') {
      extra.enableBatchRequests = info.additionalAttrs.enableBatchRequests === 'true';
    }

    // Flag custom / non-standard root tags
    const knownRootTags = [
      'api', 'sequence', 'localEntry', 'endpoint', 'template', 'task',
      'messageStore', 'messageProcessor', 'inboundEndpoint',
      'data', 'datasource', 'proxy',
    ];
    if (!knownRootTags.includes(info.rootTag)) {
      extra.isCustom = true;
      extra.rootTag = info.rootTag;
      extra.inferredFromPath = info.type !== info.rootTag;
    }

    return extra;
  }

  /**
   * SEMANTIC BOUNDARY DETECTION (Structure-Based Heuristics)
   * 
   * An XML element is considered a semantic boundary if:
   *   - It has identifying attributes (name, key, id, etc.), OR
   *   - Its tag name follows a sequence naming convention (e.g., inSequence, faultSequence)
   */
  private isSemanticBoundary(tagName: string, attrs: Record<string, string> = {}): boolean {
    // Sequence-like tags are always semantic boundaries (e.g., inSequence, faultSequence)
    if (isSequenceLikeTag(tagName)) return true;

    // Any element with non-internal attributes is a semantic boundary
    const attrCount = Object.keys(attrs).filter(k => !k.startsWith('#')).length;
    return attrCount > 0;
  }

  /**
   * Check if tag is the document's root resource type.
   */
  private isResourceType(tagName: string): boolean {
    return this.rootArtifact !== null && tagName === this.rootArtifact.rootTag;
  }

  /**
   * Check if tag is a mediator/action element.
   * Structural heuristic: elements with a dotted name (e.g., http.post, email.send)
   * are connector-style mediators.
   */
  private isMediatorType(tagName: string): boolean {
    return tagName.includes('.');
  }

  /**
   * Check if a tag is a leaf action element (self-closing tag or empty element).
   * Leaf elements like <respond/> are meaningful action instructions
   * that should be emitted as their own chunks.
   * Excludes parser-internal nodes like #text.
   */
  private isLeafElement(tagName: string, element: any): boolean {
    // Exclude parser-internal text nodes and processing instructions
    if (tagName.startsWith('#') || tagName.startsWith('?')) return false;
    // A leaf element is self-closing/empty: either not an array, or an empty array
    return !Array.isArray(element) || element.length === 0;
  }

  /**
   * Extract cross-artifact references from a chunk's XML content.
   * These patterns are structural XML reference conventions.
   */
  private extractReferencesFromContent(content: string): string[] {
    const refs = new Set<string>();
    let match;

    // <sequence key="Name"/> -> sequence reference
    const sequenceRefPattern = /<sequence\s+key=["']([^"']+)["']\s*\/>/g;
    while ((match = sequenceRefPattern.exec(content)) !== null) {
      refs.add(`sequence:${match[1]}`);
    }

    // configKey="Name" -> local entry reference
    const configKeyPattern = /configKey=["']([^"']+)["']/g;
    while ((match = configKeyPattern.exec(content)) !== null) {
      refs.add(`localEntry:${match[1]}`);
    }

    // <endpoint key="Name"/> -> endpoint reference
    const endpointRefPattern = /<endpoint\s+key=["']([^"']+)["']\s*\/>/g;
    while ((match = endpointRefPattern.exec(content)) !== null) {
      refs.add(`endpoint:${match[1]}`);
    }

    // <call-template target="Name"/> -> template reference
    const templateRefPattern = /<call-template\s+target=["']([^"']+)["']/g;
    while ((match = templateRefPattern.exec(content)) !== null) {
      refs.add(`template:${match[1]}`);
    }

    // useConfig="Name" -> data service config reference
    const useConfigPattern = /useConfig=["']([^"']+)["']/g;
    while ((match = useConfigPattern.exec(content)) !== null) {
      refs.add(`config:${match[1]}`);
    }

    // <call-query href="Name"> -> data service query reference
    const callQueryPattern = /<call-query\s+href=["']([^"']+)["']/g;
    while ((match = callQueryPattern.exec(content)) !== null) {
      refs.add(`query:${match[1]}`);
    }

    return Array.from(refs);
  }

  /**
   * EXCLUSIVE TOP-DOWN CHUNKING with token gating
   * 
   * Traverses the XML tree top-down. At each node:
   *   - If it's a semantic boundary (has attributes) -> check token limit
   *     - Fits -> emit chunk, stop descending
   *     - Too large -> descend into children
   *   - If not a boundary -> just traverse children
   */
  private processNode(
    node: any,
    lines: string[],
    filePath: string,
    chunks: XMLChunk[],
    parentChunkId: number | null,
    context: SemanticContext
  ): void {
    if (!Array.isArray(node)) return;

    for (const item of node) {
      const tagName = Object.keys(item).find(key => key !== ':@') || '';
      if (!tagName) continue;

      const element = item[tagName];
      const nodeAttrs = item[':@'] || {};

      // Update context based on node structure (attribute-driven)
      const updatedContext = this.updateContext(tagName, nodeAttrs, context);

      // Check if this is a chunkable node using structural heuristics
      const isChunkable = this.isResourceType(tagName) ||
        this.isSemanticBoundary(tagName, nodeAttrs) ||
        this.isMediatorType(tagName) ||
        this.isLeafElement(tagName, element);

      if (isChunkable) {
        // Token gating: Check if subtree fits within limit
        const range = this.findElementRange(tagName, lines);
        const content = this.extractContent(lines, range);
        const metadata = this.formatMetadata(updatedContext);
        const tokenCount = this.estimateTokenCount(content, metadata);

        if (tokenCount <= this.maxTokens) {
          // Subtree fits -> Emit chunk and STOP traversal
          this.createChunk(tagName, nodeAttrs, content, range, filePath, chunks, parentChunkId, updatedContext);
        } else {
          // Subtree too large -> Do NOT chunk, descend to ALL children
          if (Array.isArray(element)) {
            this.processNode(element, lines, filePath, chunks, parentChunkId, updatedContext);
          }
        }
      } else if (Array.isArray(element)) {
        // Non-chunkable nodes -> just traverse
        this.processNode(element, lines, filePath, chunks, parentChunkId, updatedContext);
      }
    }
  }

  /**
   * Update semantic context as we traverse the tree.
   * 
   * Uses STRUCTURAL ATTRIBUTE PATTERNS rather than hardcoded tag names:
   *   - Element with 'name' + 'context' attrs -> API-like context
   *   - Element with method + URI attrs -> resource context
   *   - Tag name ending in 'Sequence' -> sequence context (as string)
   *   - Element with 'key' attr and tag is 'sequence' -> sequence reference
   *   - Element with 'id' + 'useConfig' attrs -> query context
   *   - Element named 'operation' with 'name' attr -> operation context
   */
  private updateContext(tagName: string, attrs: Record<string, string>, parentContext: SemanticContext): SemanticContext {
    const newContext = { ...parentContext };

    const name = attrs.name || attrs['@_name'];
    const context = attrs.context || attrs['@_context'];
    const key = attrs.key || attrs['@_key'];
    const xmlns = attrs.xmlns || attrs['@_xmlns'];
    const methods = attrs.methods || attrs['@_methods'];
    const uriTemplate = attrs['uri-template'] || attrs['@_uri-template'] || attrs.uri || attrs['@_uri'];
    const id = attrs.id || attrs['@_id'];
    const useConfig = attrs.useConfig || attrs['@_useConfig'];

    // Pattern: element with both 'name' and 'context' attributes -> API-like context
    if (name && context) {
      newContext.api = { name, context, xmlns };
    }
    // Pattern: element named 'resource' with method + URI attributes -> resource context
    else if (tagName === 'resource' && methods && uriTemplate) {
      newContext.resource = { method: methods, uriTemplate };
    }
    // Pattern: tag name ends with 'Sequence' (e.g., inSequence, outSequence, faultSequence)
    else if (isSequenceLikeTag(tagName)) {
      newContext.sequence = tagName;
    }
    // Pattern: 'sequence' tag with a 'key' attribute -> sequence reference
    else if (tagName === 'sequence' && key) {
      newContext.sequence = key;
    }
    // Pattern: element with 'id' + 'useConfig' attributes -> query-like context
    else if (id && useConfig) {
      newContext.query = { id, useConfig };
    }
    // Pattern: element named 'operation' with a 'name' attribute -> operation-like context
    else if (tagName === 'operation' && name) {
      newContext.operation = { name };
    }

    return newContext;
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
    parentChunkId: number | null,
    context: SemanticContext
  ): void {
    const resourceName = attrs.name || attrs['@_name'] || attrs.key || attrs['@_key'] ||
      attrs.context || attrs['@_context'] || tagName;
    const chunkIndex = this.nextChunkIndex++;

    const embeddingText = this.buildEmbeddingText(tagName, resourceName, content, attrs, context);
    const semanticType = this.mapToSemanticType(tagName);
    const semanticIntent = this.inferIntent(tagName, attrs, content);
    const contentHash = computeChunkHash(content, {
      type: semanticType,
      intent: semanticIntent,
      context,
    });

    // Extract cross-artifact references from this chunk's content
    const chunkReferences = this.extractReferencesFromContent(content);

    // Detect if this is a standalone artifact definition
    // Structural heuristic: named elements that define reusable configuration units
    const isStandalone = this.isStandaloneDefinition(tagName);
    const sequenceKey = isStandalone ? (attrs.name || attrs['@_name'] || attrs.key || attrs['@_key']) : undefined;

    // Determine resource type from context
    const resourceType = context.artifact?.type ||
      (this.isResourceType(tagName) ? this.rootArtifact?.type || tagName : this.inferTypeFromPath(filePath));

    chunks.push({
      filePath,
      resourceName,
      resourceType,
      chunkType: tagName,
      chunkIndex,
      startLine: range.start,
      endLine: range.end,
      content,
      parentChunkId,
      embeddingText,
      semanticType,
      semanticIntent,
      contentHash,
      context: { ...context, references: chunkReferences.length > 0 ? chunkReferences : undefined },
      sequenceKey,
      isSequenceDefinition: isStandalone,
      referencedSequences: chunkReferences,
    });
  }

  /**
   * Detect if an element is a standalone artifact definition.
   * Elements like sequence, localEntry, endpoint, template are reusable units.
   */
  private isStandaloneDefinition(tagName: string): boolean {
    const standalonePatterns = ['sequence', 'localEntry', 'endpoint', 'template'];
    return standalonePatterns.includes(tagName);
  }

  /**
   * Map XML tag to semantic type using structural naming patterns.
   */
  private mapToSemanticType(tagName: string): string {
    if (isSequenceLikeTag(tagName) || tagName === 'sequence') return 'sequence';

    const typeMap: Record<string, string> = {
      resource:        'resource',
      api:             'api',
      proxy:           'api',
      filter:          'filter',
      switch:          'filter',
      payloadFactory:  'payloadFactory',
      respond:         'response',
      config:          'dataConfig',
      query:           'dataQuery',
      operation:       'dataOperation',
      trigger:         'trigger',
      property:        'property',
    };

    return typeMap[tagName] || 'component';
  }

  /**
   * Infer semantic intent from tag structure and content.
   */
  private inferIntent(tagName: string, attrs: Record<string, string>, content: string): string {
    if (tagName === 'filter' || tagName === 'switch') return 'validation';
    if (tagName === 'payloadFactory' || tagName === 'enrich') return 'transformation';
    if (tagName === 'call' || tagName === 'send' || tagName.startsWith('http.')) return 'delegation';
    if (tagName === 'respond') return 'response';
    if (isSequenceLikeTag(tagName) && tagName.toLowerCase().includes('fault')) return 'error-handling';
    if (tagName === 'query' || tagName === 'operation') return 'data-access';
    if (tagName === 'config' || tagName === 'property' || tagName === 'trigger') return 'configuration';

    return 'processing';
  }

  /**
   * Estimate token count using the loaded AutoTokenizer.
   * Falls back to character-based approximation if tokenizer unavailable.
   */
  private estimateTokenCount(content: string, metadata: string = ''): number {
    const fullText = metadata + ' ' + content;

    if (this.tokenizer) {
      const encoded = this.tokenizer.encode(fullText);
      return encoded.length;
    }

    return Math.ceil(fullText.length / 4);
  }

  /**
   * Format context metadata into text for token counting and embedding prefix.
   */
  private formatMetadata(context: SemanticContext): string {
    const parts: string[] = [];
    if (context.api?.name) parts.push(`API: ${context.api.name}`);
    if (context.api?.context) parts.push(`Context: ${context.api.context}`);
    if (context.resource?.method) parts.push(`Method: ${context.resource.method}`);
    if (context.resource?.uriTemplate) parts.push(`URI: ${context.resource.uriTemplate}`);
    if (context.sequence) {
      const seqName = typeof context.sequence === 'string' ? context.sequence : context.sequence.name;
      parts.push(`Sequence: ${seqName}`);
    }
    if (context.artifact?.name) parts.push(`${context.artifact.type}: ${context.artifact.name}`);
    if (context.query?.id) parts.push(`Query: ${context.query.id}`);
    if (context.operation?.name) parts.push(`Operation: ${context.operation.name}`);
    if (context.references && context.references.length > 0) {
      parts.push(`Uses: ${context.references.join(', ')}`);
    }
    return parts.join(' ');
  }

  private findElementRange(tagName: string, lines: string[]): LineRange {
    let startLine = -1;
    let endLine = -1;
    let depth = 0;

    for (let i = this.lineSearchCursor; i < lines.length; i++) {
      const line = lines[i];

      if (startLine === -1) {
        const openPattern = new RegExp(`<${tagName}[\\s>/]`);
        if (openPattern.test(line)) {
          startLine = i + 1;
          this.lineSearchCursor = i + 1;

          if (line.includes('/>')) {
            endLine = i + 1;
            break;
          }
          depth = 1;
        }
      } else {
        const openPattern = new RegExp(`<${tagName}[\\s>/]`);
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
   * Infer resource type from file path as a fallback.
   */
  private inferTypeFromPath(filePath: string): string {
    const folderPatterns: [RegExp, string][] = [
      [/\/apis\//,                'api'],
      [/\/sequences\//,           'sequence'],
      [/\/proxy-services\//,      'proxy'],
      [/\/endpoints\//,           'endpoint'],
      [/\/local-entries\//,       'localEntry'],
      [/\/templates\//,           'template'],
      [/\/data-services\//,       'dataService'],
      [/\/data-sources\//,        'dataSource'],
      [/\/tasks\//,               'task'],
      [/\/message-stores\//,      'messageStore'],
      [/\/message-processors\//,  'messageProcessor'],
      [/\/inbound-endpoints\//,   'inboundEndpoint'],
    ];

    for (const [pattern, type] of folderPatterns) {
      if (pattern.test(filePath)) return type;
    }

    return 'unknown';
  }

  /**
   * Build natural text representation for embedding.
   * Format: [Context Metadata] + [Cleaned XML Content]
   */
  private buildEmbeddingText(
    tagName: string,
    resourceName: string,
    content: string,
    attrs: Record<string, string>,
    context: SemanticContext
  ): string {
    const contextStr = this.formatMetadata(context);
    const tokens: string[] = contextStr ? [contextStr] : [];

    const cleanedContent = content
      .replace(/<([^>\/\s]+)([^>]*)>/g, ' $1 $2 ')
      .replace(/<\/[^>]+>/g, ' ')
      .replace(/<([^>\/\s]+)([^>]*)\s*\/>/g, ' $1 $2 ')
      .replace(/="([^"]*)"/g, '=$1')
      .replace(/='([^']*)'/g, '=$1')
      .replace(/[^\w\s=\$\{\}\[\]\/\-\.,:@]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    const contentTokens = cleanedContent
      .split(/\s+/)
      .filter(t => t.length > 1 && t.length < 100);

    tokens.push(...contentTokens);

    return tokens.join(' ');
  }
}
