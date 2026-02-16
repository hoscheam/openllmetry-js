import { context, Histogram, metrics } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  BatchSpanProcessor,
  ReadableSpan,
  SimpleSpanProcessor,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import {
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_CONVERSATION_ID,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
} from "@opentelemetry/semantic-conventions/incubating";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  transformAiSdkSpanAttributes,
  transformAiSdkSpanNames,
} from "./ai-sdk-transformations";
import { parseKeyPairsIntoRecord } from "./baggage-utils";
import {
  AGENT_NAME_KEY,
  ASSOCATION_PROPERTIES_KEY,
  CONVERSATION_ID_KEY,
  ENTITY_NAME_KEY,
  WORKFLOW_NAME_KEY,
} from "./tracing";

export const ALL_INSTRUMENTATION_LIBRARIES = "all" as const;
type AllInstrumentationLibraries = typeof ALL_INSTRUMENTATION_LIBRARIES;

// Metric name for prompt caching (Dynatrace compatibility)
const METRIC_GEN_AI_PROMPT_CACHING = "gen_ai.prompt.caching";

// Lazy-initialized histogram for prompt caching metrics
let promptCachingHistogram: Histogram | undefined;

const getPromptCachingHistogram = (): Histogram => {
  if (!promptCachingHistogram) {
    const meter = metrics.getMeter("@traceloop/node-server-sdk");
    promptCachingHistogram = meter.createHistogram(
      METRIC_GEN_AI_PROMPT_CACHING,
      {
        description:
          "Measures number of tokens used for prompt caching (read/create)",
        unit: "{token}",
      },
    );
  }
  return promptCachingHistogram;
};

const spanAgentNames = new Map<
  string,
  { agentName: string; timestamp: number }
>();

const SPAN_AGENT_NAME_TTL = 5 * 60 * 1000;

const AI_TELEMETRY_METADATA_AGENT = "ai.telemetry.metadata.agent";

const cleanupExpiredSpanAgentNames = (): void => {
  const now = Date.now();
  for (const [spanId, entry] of spanAgentNames.entries()) {
    if (now - entry.timestamp > SPAN_AGENT_NAME_TTL) {
      spanAgentNames.delete(spanId);
    }
  }
};

export interface SpanProcessorOptions {
  /**
   * The API Key for sending traces data. Optional.
   * Defaults to the TRACELOOP_API_KEY environment variable.
   */
  apiKey?: string;

  /**
   * The OTLP endpoint for sending traces data. Optional.
   * Defaults to TRACELOOP_BASE_URL environment variable or https://api.traceloop.com/
   */
  baseUrl?: string;

  /**
   * Sends traces and spans without batching, for local development. Optional.
   * Defaults to false.
   */
  disableBatch?: boolean;

  /**
   * The OpenTelemetry SpanExporter to be used for sending traces data. Optional.
   * Defaults to the OTLP exporter.
   */
  exporter?: SpanExporter;

  /**
   * The headers to be sent with the traces data. Optional.
   */
  headers?: Record<string, string>;

  /**
   * The instrumentation libraries to be allowed in the traces. Optional.
   * Defaults to Traceloop instrumentation libraries.
   * If set to ALL_INSTRUMENTATION_LIBRARIES, all instrumentation libraries will be allowed.
   * If set to an array of instrumentation libraries, only traceloop instrumentation libraries and the specified instrumentation libraries will be allowed.
   */
  allowedInstrumentationLibraries?: string[] | AllInstrumentationLibraries;
}

/**
 * Creates a span processor with Traceloop's custom span handling logic.
 * This can be used independently of the full SDK initialization.
 *
 * @param options - Configuration options for the span processor
 * @returns A configured SpanProcessor instance
 */
export const createSpanProcessor = (
  options: SpanProcessorOptions,
): SpanProcessor => {
  const url = `${options.baseUrl || process.env.TRACELOOP_BASE_URL || "https://api.traceloop.com"}/v1/traces`;
  const headers =
    options.headers ||
    (process.env.TRACELOOP_HEADERS
      ? parseKeyPairsIntoRecord(process.env.TRACELOOP_HEADERS)
      : { Authorization: `Bearer ${options.apiKey}` });

  const traceExporter =
    options.exporter ??
    new OTLPTraceExporter({
      url,
      headers,
    });

  const spanProcessor = options.disableBatch
    ? new SimpleSpanProcessor(traceExporter)
    : new BatchSpanProcessor(traceExporter);

  // Store the original onEnd method
  const originalOnEnd = spanProcessor.onEnd.bind(spanProcessor);

  spanProcessor.onStart = onSpanStart;

  if (
    options.allowedInstrumentationLibraries === ALL_INSTRUMENTATION_LIBRARIES
  ) {
    spanProcessor.onEnd = onSpanEnd(originalOnEnd);
  } else {
    const instrumentationLibraries = [...traceloopInstrumentationLibraries];

    if (options.allowedInstrumentationLibraries) {
      instrumentationLibraries.push(...options.allowedInstrumentationLibraries);
    }

    spanProcessor.onEnd = onSpanEnd(originalOnEnd, instrumentationLibraries);
  }

  return spanProcessor;
};

export const traceloopInstrumentationLibraries = [
  "ai",
  "@traceloop/node-server-sdk",
  "@traceloop/instrumentation-openai",
  "@traceloop/instrumentation-langchain",
  "@traceloop/instrumentation-chroma",
  "@traceloop/instrumentation-anthropic",
  "@traceloop/instrumentation-llamaindex",
  "@traceloop/instrumentation-vertexai",
  "@traceloop/instrumentation-bedrock",
  "@traceloop/instrumentation-cohere",
  "@traceloop/instrumentation-pinecone",
  "@traceloop/instrumentation-qdrant",
  "@traceloop/instrumentation-together",
  "@traceloop/instrumentation-mcp",
];

const onSpanStart = (span: Span): void => {
  const workflowName = context.active().getValue(WORKFLOW_NAME_KEY);
  if (workflowName) {
    span.setAttribute(
      SpanAttributes.TRACELOOP_WORKFLOW_NAME,
      workflowName as string,
    );
  }

  const entityName = context.active().getValue(ENTITY_NAME_KEY);
  if (entityName) {
    span.setAttribute(
      SpanAttributes.TRACELOOP_ENTITY_PATH,
      entityName as string,
    );
  }

  let agentName = context.active().getValue(AGENT_NAME_KEY) as
    | string
    | undefined;

  if (!agentName) {
    const aiSdkAgent = span.attributes[AI_TELEMETRY_METADATA_AGENT];
    if (aiSdkAgent && typeof aiSdkAgent === "string") {
      agentName = aiSdkAgent;
    }
  }

  if (!agentName) {
    const parentSpanContext = (span as any).parentSpanContext;
    const parentSpanId = parentSpanContext?.spanId;
    if (
      parentSpanId &&
      parentSpanId !== "0000000000000000" &&
      spanAgentNames.has(parentSpanId)
    ) {
      agentName = spanAgentNames.get(parentSpanId)!.agentName;
    }
  }

  if (agentName) {
    span.setAttribute(ATTR_GEN_AI_AGENT_NAME, agentName);
    const spanId = span.spanContext().spanId;
    spanAgentNames.set(spanId, { agentName, timestamp: Date.now() });
  }

  // Check for conversation ID in context
  const conversationId = context.active().getValue(CONVERSATION_ID_KEY);
  if (conversationId) {
    span.setAttribute(ATTR_GEN_AI_CONVERSATION_ID, conversationId as string);
  }

  // Check for association properties in context (set by decorators or withAssociationProperties)
  const associationProperties = context
    .active()
    .getValue(ASSOCATION_PROPERTIES_KEY) as
    | { [name: string]: string }
    | undefined;

  if (associationProperties && Object.keys(associationProperties).length > 0) {
    for (const [key, value] of Object.entries(associationProperties)) {
      span.setAttribute(
        `${SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.${key}`,
        value,
      );
    }
  }

  transformAiSdkSpanNames(span);
};

/**
 * Ensures span compatibility between OTel v1.x and v2.x for OTLP transformer
 */
const ensureSpanCompatibility = (span: ReadableSpan): ReadableSpan => {
  const spanAny = span as any;

  // If the span already has instrumentationLibrary, it's compatible (OTel v2.x)
  if (spanAny.instrumentationLibrary) {
    return span;
  }

  // If it has instrumentationScope but no instrumentationLibrary (OTel v1.x),
  // add instrumentationLibrary as an alias to prevent OTLP transformer errors
  if (spanAny.instrumentationScope) {
    // Create a proxy that provides both properties
    return new Proxy(span, {
      get(target, prop) {
        if (prop === "instrumentationLibrary") {
          return (target as any).instrumentationScope;
        }
        return (target as any)[prop];
      },
    }) as ReadableSpan;
  }

  // Fallback: add both properties with defaults
  return new Proxy(span, {
    get(target, prop) {
      if (
        prop === "instrumentationLibrary" ||
        prop === "instrumentationScope"
      ) {
        return {
          name: "unknown",
          version: undefined,
          schemaUrl: undefined,
        };
      }
      return (target as any)[prop];
    },
  }) as ReadableSpan;
};

const onSpanEnd = (
  originalOnEnd: (span: ReadableSpan) => void,
  instrumentationLibraries?: string[],
) => {
  return (span: ReadableSpan): void => {
    if (
      instrumentationLibraries &&
      !instrumentationLibraries.includes(
        (span as any).instrumentationScope?.name ||
          (span as any).instrumentationLibrary?.name,
      )
    ) {
      return;
    }

    transformAiSdkSpanAttributes(span);

    // Record prompt caching metrics for Dynatrace compatibility
    recordPromptCachingMetrics(span);

    const spanId = span.spanContext().spanId;
    const parentSpanId = span.parentSpanContext?.spanId;
    let agentName = span.attributes[ATTR_GEN_AI_AGENT_NAME];

    if (agentName && typeof agentName === "string") {
      spanAgentNames.set(spanId, {
        agentName,
        timestamp: Date.now(),
      });
    } else if (
      parentSpanId &&
      parentSpanId !== "0000000000000000" &&
      spanAgentNames.has(parentSpanId)
    ) {
      agentName = spanAgentNames.get(parentSpanId)!.agentName;
      span.attributes[ATTR_GEN_AI_AGENT_NAME] = agentName;
      spanAgentNames.set(spanId, {
        agentName,
        timestamp: Date.now(),
      });
    }

    if (Math.random() < 0.01) {
      cleanupExpiredSpanAgentNames();
    }

    const compatibleSpan = ensureSpanCompatibility(span);

    originalOnEnd(compatibleSpan);
  };
};

/**
 * Records prompt caching metrics from AI SDK span attributes
 * Emits gen_ai.prompt.caching metric with gen_ai.cache.type dimension
 */
const recordPromptCachingMetrics = (span: ReadableSpan): void => {
  const cacheReadTokens =
    span.attributes[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS];
  const cacheCreateTokens =
    span.attributes[SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS];

  if (!cacheReadTokens && !cacheCreateTokens) {
    return;
  }

  const histogram = getPromptCachingHistogram();

  const metricAttributes: Record<string, string> = {};

  // Add gen_ai.system if available
  const genAiSystem = span.attributes[ATTR_GEN_AI_SYSTEM];
  if (genAiSystem && typeof genAiSystem === "string") {
    metricAttributes[ATTR_GEN_AI_SYSTEM] = genAiSystem;
  }

  // Add gen_ai.request.model if available
  const requestModel = span.attributes[ATTR_GEN_AI_REQUEST_MODEL];
  if (requestModel && typeof requestModel === "string") {
    metricAttributes[ATTR_GEN_AI_REQUEST_MODEL] = requestModel;
  }

  // Add gen_ai.response.model if available
  const responseModel = span.attributes[ATTR_GEN_AI_RESPONSE_MODEL];
  if (responseModel && typeof responseModel === "string") {
    metricAttributes[ATTR_GEN_AI_RESPONSE_MODEL] = responseModel;
  }

  // Record cache read tokens metric
  if (
    cacheReadTokens !== undefined &&
    cacheReadTokens !== null &&
    typeof cacheReadTokens === "number" &&
    cacheReadTokens > 0
  ) {
    histogram.record(cacheReadTokens, {
      ...metricAttributes,
      "gen_ai.cache.type": "read",
    });
  }

  // Record cache creation tokens metric
  if (
    cacheCreateTokens !== undefined &&
    cacheCreateTokens !== null &&
    typeof cacheCreateTokens === "number" &&
    cacheCreateTokens > 0
  ) {
    histogram.record(cacheCreateTokens, {
      ...metricAttributes,
      "gen_ai.cache.type": "create",
    });
  }
};
