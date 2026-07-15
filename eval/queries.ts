/**
 * Labeled retrieval queries over the artifacts/ corpus.
 *
 * Each query has one or more gold spans: (file, inclusive line range) pairs
 * that contain the information needed to answer it. A retrieved chunk is
 * relevant iff it comes from a gold file and its line span overlaps the gold
 * range. Queries deliberately mix vocabulary that appears verbatim in the
 * configs with paraphrases that do not (e.g. "system prompt", "JDBC driver",
 * "rate limit"), reflecting how integration developers actually ask.
 *
 * Line numbers refer to the files as committed in this repository.
 */

export interface GoldSpan {
  file: string;
  lines: [number, number];
}

export interface EvalQuery {
  id: string;
  query: string;
  gold: GoldSpan[];
}

const API_BANK = 'artifacts/apis/BankAPI.xml';
const API_AI = 'artifacts/apis/AIAgentAPI.xml';
const DS_BANK = 'artifacts/data-services/BankDataService.xml';
const SEQ_AUTH = 'artifacts/sequences/AuthenticationSequence.xml';
const SEQ_ERROR = 'artifacts/sequences/GlobalErrorSequence.xml';
const SEQ_LOG = 'artifacts/sequences/LoggingSequence.xml';
const SEQ_NOTIFY = 'artifacts/sequences/NotificationSequence.xml';
const SEQ_RATE = 'artifacts/sequences/RateLimitSequence.xml';
const LE_CURRENCY = 'artifacts/local-entries/CurrencyConverter.xml';
const LE_EMAIL = 'artifacts/local-entries/EmailConnection.xml';
const LE_FILEMEM = 'artifacts/local-entries/FileMemoryConnection.xml';
const LE_OPENAI = 'artifacts/local-entries/OpenAIConnection.xml';
const TPL_BALANCE = 'artifacts/templates/GetAccountBalanceTool.xml';
const TPL_HISTORY = 'artifacts/templates/GetTransactionHistoryTool.xml';

export const queries: EvalQuery[] = [
  // ---- BankAPI ----
  {
    id: 'bank-deposit-conversion',
    query: 'How does the bank API convert deposited amounts to USD?',
    gold: [{ file: API_BANK, lines: [14, 60] }],
  },
  {
    id: 'bank-greeting',
    query: 'Which endpoint returns a welcome greeting message?',
    gold: [{ file: API_BANK, lines: [3, 12] }],
  },
  {
    id: 'bank-balance-endpoint',
    query: 'How is the account balance returned for a given account ID in the REST API?',
    gold: [{ file: API_BANK, lines: [62, 78] }],
  },
  {
    id: 'bank-transfer-validation',
    query: 'What happens when a transfer amount is zero or negative?',
    gold: [{ file: API_BANK, lines: [86, 106] }],
  },
  {
    id: 'bank-deposit-fault',
    query: 'How does the API respond when a deposit fails?',
    gold: [{ file: API_BANK, lines: [48, 59] }],
  },
  {
    id: 'bank-currency-service',
    query: 'Which external service is called to get currency exchange rates?',
    gold: [
      { file: API_BANK, lines: [19, 29] },
      { file: LE_CURRENCY, lines: [2, 14] },
    ],
  },
  {
    id: 'bank-transactions-endpoint',
    query: 'How can I see the recent transactions for an account via the API?',
    gold: [{ file: API_BANK, lines: [122, 149] }],
  },
  {
    id: 'bank-deposit-response',
    query: 'How is the deposit response payload structured after currency conversion?',
    gold: [{ file: API_BANK, lines: [34, 46] }],
  },
  {
    id: 'bank-transfer-response',
    query: 'What details are included in a successful transfer confirmation?',
    gold: [{ file: API_BANK, lines: [87, 102] }],
  },

  // ---- AIAgentAPI ----
  {
    id: 'ai-instructions',
    query: 'What system prompt or instructions does the banking AI assistant use?',
    gold: [{ file: API_AI, lines: [14, 17] }],
  },
  {
    id: 'ai-model-params',
    query: 'Which LLM model and generation parameters does the AI agent use?',
    gold: [{ file: API_AI, lines: [20, 24] }],
  },
  {
    id: 'ai-tools',
    query: 'What tools can the AI agent invoke?',
    gold: [{ file: API_AI, lines: [25, 29] }],
  },
  {
    id: 'ai-memory',
    query: 'How does the AI agent maintain conversation memory between requests?',
    gold: [
      { file: API_AI, lines: [10, 14] },
      { file: LE_FILEMEM, lines: [2, 7] },
    ],
  },
  {
    id: 'ai-openai-config',
    query: 'How is the OpenAI connection configured, including the API key?',
    gold: [{ file: LE_OPENAI, lines: [2, 9] }],
  },
  {
    id: 'ai-health',
    query: 'What is returned by the AI agent health check endpoint?',
    gold: [{ file: API_AI, lines: [62, 75] }],
  },
  {
    id: 'ai-error-handling',
    query: 'How does the AI agent API respond when processing a chat request fails?',
    gold: [{ file: API_AI, lines: [47, 59] }],
  },

  // ---- BankDataService ----
  {
    id: 'ds-db-config',
    query: 'Which database and JDBC driver does the bank data service use?',
    gold: [{ file: DS_BANK, lines: [2, 7] }],
  },
  {
    id: 'ds-balance-query',
    query: 'What SQL query fetches the account balance?',
    gold: [{ file: DS_BANK, lines: [9, 16] }],
  },
  {
    id: 'ds-insert-transaction',
    query: 'How are new transactions inserted into the database?',
    gold: [{ file: DS_BANK, lines: [24, 31] }],
  },
  {
    id: 'ds-history-query',
    query: 'Which SQL query returns the last ten transactions for an account?',
    gold: [{ file: DS_BANK, lines: [33, 43] }],
  },
  {
    id: 'ds-update-operation',
    query: 'How is updating an account balance exposed as a data service operation?',
    gold: [
      { file: DS_BANK, lines: [18, 22] },
      { file: DS_BANK, lines: [51, 56] },
    ],
  },

  // ---- Sequences ----
  {
    id: 'auth-bearer',
    query: 'How does the system validate Bearer tokens in the Authorization header?',
    gold: [{ file: SEQ_AUTH, lines: [3, 21] }],
  },
  {
    id: 'auth-missing-header',
    query: 'What error is raised when the Authorization header is missing?',
    gold: [{ file: SEQ_AUTH, lines: [18, 20] }],
  },
  {
    id: 'error-auth-status',
    query: 'What HTTP status code is returned for authentication failures?',
    gold: [{ file: SEQ_ERROR, lines: [12, 22] }],
  },
  {
    id: 'error-validation-status',
    query: 'How are validation errors mapped to HTTP error responses?',
    gold: [{ file: SEQ_ERROR, lines: [23, 33] }],
  },
  {
    id: 'error-default',
    query: 'What is the default error response for unknown error codes?',
    gold: [{ file: SEQ_ERROR, lines: [34, 44] }],
  },
  {
    id: 'log-db-insert',
    query: 'How are API requests logged to the database?',
    gold: [{ file: SEQ_LOG, lines: [11, 26] }],
  },
  {
    id: 'log-variables',
    query: 'What request details are captured before logging each API call?',
    gold: [{ file: SEQ_LOG, lines: [3, 9] }],
  },
  {
    id: 'notify-email',
    query: 'How are customers notified about their transactions by email?',
    gold: [{ file: SEQ_NOTIFY, lines: [7, 23] }],
  },
  {
    id: 'email-smtp',
    query: 'Which SMTP server and port are used for sending emails?',
    gold: [{ file: LE_EMAIL, lines: [2, 12] }],
  },
  {
    id: 'rate-limits',
    query: 'What rate limits are applied to API clients?',
    gold: [{ file: SEQ_RATE, lines: [7, 26] }],
  },
  {
    id: 'rate-reject',
    query: 'What response does a client get when exceeding the rate limit?',
    gold: [{ file: SEQ_RATE, lines: [32, 40] }],
  },
  {
    id: 'rate-missing-client',
    query: 'What happens when the X-Client-ID header is missing from a request?',
    gold: [
      { file: SEQ_RATE, lines: [3, 5] },
      { file: SEQ_RATE, lines: [43, 45] },
    ],
  },

  // ---- Templates ----
  {
    id: 'tpl-balance-tool',
    query: 'Which template retrieves the balance for the AI agent tool calls?',
    gold: [{ file: TPL_BALANCE, lines: [2, 15] }],
  },
  {
    id: 'tpl-history-tool',
    query: 'What does the transaction history tool template return?',
    gold: [{ file: TPL_HISTORY, lines: [5, 25] }],
  },

  // ---- Local entries ----
  {
    id: 'currency-timeout',
    query: 'Where are the timeout and retry settings for the currency HTTP connection defined?',
    gold: [{ file: LE_CURRENCY, lines: [7, 11] }],
  },
];
