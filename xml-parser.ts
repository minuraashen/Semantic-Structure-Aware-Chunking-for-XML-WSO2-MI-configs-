import * as sax from 'sax';

/**
 * Position-annotated XML tree built on the battle-tested `sax` parser.
 *
 * Every element node carries its exact character offsets in the source text,
 * so the chunker derives BOTH structure and content from one representation.
 * This eliminates the tree/text alignment problem of the previous design,
 * which parsed with fast-xml-parser (no position tracking) and then had to
 * re-locate elements in the raw text with line-based regex searches — fragile
 * against nested same-name tags, multi-line opening tags, and regex
 * metacharacters in tag names such as `http.post`.
 *
 * sax handles the parsing edge cases (comments, CDATA, processing
 * instructions, DOCTYPE, entity decoding, quoted attributes) and reports
 * `startTagPosition` / `position` while parsing, which we record on each node.
 */

export interface XMLElementNode {
  kind: 'element';
  /** Tag name exactly as written, including any namespace prefix. */
  tag: string;
  /** Attributes with entity-decoded values (sax decodes them). */
  attrs: Record<string, string>;
  children: XMLNode[];
  selfClosing: boolean;
  /** Offset of the opening `<`. */
  startOffset: number;
  /** Offset just past the final `>` (of the closing tag, or the self-closing tag). */
  endOffset: number;
}

export interface XMLTextNode {
  kind: 'text';
  /** Entity-decoded text content (for CDATA: the literal section content). */
  text: string;
  cdata: boolean;
}

export type XMLNode = XMLElementNode | XMLTextNode;

export interface XMLDocument {
  source: string;
  roots: XMLNode[];
  /** 1-based line number of a character offset. */
  lineOf(offset: number): number;
}

/** Local part of a possibly namespace-prefixed tag name (wsp:Policy → Policy). */
export function localName(tag: string): string {
  const idx = tag.lastIndexOf(':');
  return idx === -1 ? tag : tag.slice(idx + 1);
}

/** Element children of a node, skipping text nodes. */
export function childElements(el: XMLElementNode): XMLElementNode[] {
  return el.children.filter((c): c is XMLElementNode => c.kind === 'element');
}

/**
 * Parse XML into a position-annotated tree.
 * Throws on malformed input (sax strict mode), with line/column in the message.
 */
export function parseXML(source: string): XMLDocument {
  const parser = sax.parser(true /* strict */, { position: true });

  const roots: XMLNode[] = [];
  const stack: XMLElementNode[] = [];
  const addNode = (node: XMLNode): void => {
    (stack.length ? stack[stack.length - 1].children : roots).push(node);
  };

  // CDATA arrives via onopencdata/oncdata(chunks)/onclosecdata — accumulate.
  let cdataBuffer: string | null = null;

  parser.onopentag = (tag) => {
    const el: XMLElementNode = {
      kind: 'element',
      tag: tag.name,
      attrs: { ...(tag.attributes as Record<string, string>) },
      children: [],
      selfClosing: !!(tag as sax.Tag).isSelfClosing,
      // sax's startTagPosition is 1-based (position just past '<'), so the
      // zero-based offset of '<' is startTagPosition - 1. Verified by the
      // round-trip invariant in the unit tests.
      startOffset: parser.startTagPosition - 1,
      // Provisional: end of the opening tag; finalized in onclosetag.
      endOffset: parser.position,
    };
    addNode(el);
    stack.push(el);
  };

  parser.onclosetag = () => {
    const el = stack.pop();
    if (el) el.endOffset = parser.position;
  };

  parser.ontext = (text) => {
    if (text.trim().length > 0) {
      addNode({ kind: 'text', text, cdata: false });
    }
  };

  parser.onopencdata = () => {
    cdataBuffer = '';
  };
  parser.oncdata = (text) => {
    cdataBuffer = (cdataBuffer ?? '') + text;
  };
  parser.onclosecdata = () => {
    if (cdataBuffer !== null && cdataBuffer.length > 0) {
      addNode({ kind: 'text', text: cdataBuffer, cdata: true });
    }
    cdataBuffer = null;
  };

  parser.onerror = (err) => {
    // Fail fast on malformed XML — a silent wrong parse would corrupt chunks.
    throw err;
  };

  parser.write(source).close();

  // Precompute line starts for offset→line mapping (1-based lines).
  const lineStarts: number[] = [0];
  for (let i = 0; i < source.length; i++) {
    if (source[i] === '\n') lineStarts.push(i + 1);
  }
  const lineOf = (offset: number): number => {
    let lo = 0;
    let hi = lineStarts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (lineStarts[mid] <= offset) lo = mid;
      else hi = mid - 1;
    }
    return lo + 1;
  };

  return { source, roots, lineOf };
}
