/*
 * Copyright Traceloop
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { Serialized } from "@langchain/core/load/serializable";
import { BaseMessage } from "@langchain/core/messages";
import { LLMResult } from "@langchain/core/outputs";
import { ChainValues } from "@langchain/core/utils/types";
import { SpanKind, SpanStatusCode, Tracer } from "@opentelemetry/api";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

interface SpanData {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  span: any;
  runId: string;
  startTime: number;
  firstTokenTime?: number;
  // Store model name from the LLM object passed to handleChatModelStart
  modelFromConfig?: string;
}

export class TraceloopCallbackHandler extends BaseCallbackHandler {
  name = "traceloop_callback_handler";

  private tracer: Tracer;
  private spans: Map<string, SpanData> = new Map();
  private traceContent: boolean;

  constructor(tracer: Tracer, traceContent = true) {
    super();
    this.tracer = tracer;
    this.traceContent = traceContent;
  }

  override async handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    _parentRunId?: string,
    _extraParams?: Record<string, unknown>,
    _tags?: string[],
    _metadata?: Record<string, unknown>,
    _runName?: string,
  ): Promise<void> {
    const startTime = Date.now();

    // Use detected class name which checks multiple sources for accurate detection
    const detectedClassName = this.detectClassName(llm);
    const className =
      detectedClassName || llm.id?.[llm.id.length - 1] || "unknown";
    const vendor = this.detectVendor(llm);
    const spanBaseName = this.convertClassNameToSpanName(className);

    // Try to extract model name from the LLM config at start time
    const modelFromConfig = this.extractModelNameFromConfig(llm);

    // Create single LLM span like Python implementation
    const span = this.tracer.startSpan(spanBaseName, {
      kind: SpanKind.CLIENT,
    });

    const flatMessages = messages.flat();
    span.setAttributes({
      [ATTR_GEN_AI_SYSTEM]: vendor,
      [SpanAttributes.LLM_REQUEST_TYPE]: "chat",
    });

    // Add prompts if tracing content
    if (this.traceContent && flatMessages.length > 0) {
      flatMessages.forEach((message, idx) => {
        const role = this.mapMessageTypeToRole(message._getType());
        span.setAttributes({
          [`${ATTR_GEN_AI_PROMPT}.${idx}.role`]: role,
          [`${ATTR_GEN_AI_PROMPT}.${idx}.content`]:
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message.content),
        });
      });
    }

    this.spans.set(runId, {
      span,
      runId,
      startTime,
      modelFromConfig: modelFromConfig || undefined,
    });
  }

  override async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    _parentRunId?: string,
    _extraParams?: Record<string, unknown>,
    _tags?: string[],
    _metadata?: Record<string, unknown>,
    _runName?: string,
  ): Promise<void> {
    const startTime = Date.now();

    // Use detected class name which checks multiple sources for accurate detection
    const detectedClassName = this.detectClassName(llm);
    const className =
      detectedClassName || llm.id?.[llm.id.length - 1] || "unknown";
    const vendor = this.detectVendor(llm);
    const spanBaseName = this.convertClassNameToSpanName(className);

    // Try to extract model name from the LLM config at start time
    const modelFromConfig = this.extractModelNameFromConfig(llm);

    // Create single LLM span like handleChatModelStart
    const span = this.tracer.startSpan(spanBaseName, {
      kind: SpanKind.CLIENT,
    });

    span.setAttributes({
      [ATTR_GEN_AI_SYSTEM]: vendor,
      [SpanAttributes.LLM_REQUEST_TYPE]: "completion",
    });

    if (this.traceContent && prompts.length > 0) {
      prompts.forEach((prompt, idx) => {
        span.setAttributes({
          [`${ATTR_GEN_AI_PROMPT}.${idx}.role`]: "user",
          [`${ATTR_GEN_AI_PROMPT}.${idx}.content`]: prompt,
        });
      });
    }

    this.spans.set(runId, {
      span,
      runId,
      startTime,
      modelFromConfig: modelFromConfig || undefined,
    });
  }

  /**
   * Handle new token for streaming - track time to first token
   */
  override async handleLLMNewToken(
    token: string,
    _idx?: unknown,
    runId?: string,
    _parentRunId?: string,
    _tags?: string[],
    _fields?: unknown,
  ): Promise<void> {
    if (!runId) return;

    const spanData = this.spans.get(runId);
    if (!spanData) return;

    // Record first token time if not already set
    if (!spanData.firstTokenTime) {
      spanData.firstTokenTime = Date.now();
      const timeToFirstToken = spanData.firstTokenTime - spanData.startTime;

      // Set time to first token attribute (in milliseconds)
      spanData.span.setAttributes({
        "gen_ai.response.time_to_first_token_ms": timeToFirstToken,
        "llm.response.time_to_first_token": timeToFirstToken / 1000, // Also in seconds
      });
    }
  }

  override async handleLLMEnd(
    output: LLMResult,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _extraParams?: Record<string, unknown>,
  ): Promise<void> {
    const spanData = this.spans.get(runId);
    if (!spanData) return;

    const { span, startTime, modelFromConfig } = spanData;
    const endTime = Date.now();
    const totalDuration = endTime - startTime;

    // Set duration/latency attributes
    span.setAttributes({
      "gen_ai.response.duration_ms": totalDuration,
      "llm.response.duration": totalDuration / 1000, // In seconds
    });

    if (
      this.traceContent &&
      output.generations &&
      output.generations.length > 0
    ) {
      output.generations.forEach((generation, idx) => {
        if (generation && generation.length > 0) {
          span.setAttributes({
            [`${ATTR_GEN_AI_COMPLETION}.${idx}.role`]: "assistant",
            [`${ATTR_GEN_AI_COMPLETION}.${idx}.content`]: generation[0].text,
          });
        }
      });
    }

    // Extract model name from response, falling back to config model
    const modelFromResponse = this.extractModelNameFromResponse(output);
    const modelName = modelFromResponse || modelFromConfig;

    // Set both request and response model attributes
    span.setAttributes({
      [ATTR_GEN_AI_REQUEST_MODEL]: modelName || "unknown",
      [ATTR_GEN_AI_RESPONSE_MODEL]: modelName || "unknown",
    });

    // Add usage metrics if available
    if (output.llmOutput?.usage) {
      const usage = output.llmOutput.usage;
      if (usage.input_tokens) {
        span.setAttributes({
          [ATTR_GEN_AI_USAGE_PROMPT_TOKENS]: usage.input_tokens,
        });
      }
      if (usage.output_tokens) {
        span.setAttributes({
          [ATTR_GEN_AI_USAGE_COMPLETION_TOKENS]: usage.output_tokens,
        });
      }
      const totalTokens =
        (usage.input_tokens || 0) + (usage.output_tokens || 0);
      if (totalTokens > 0) {
        span.setAttributes({
          [SpanAttributes.LLM_USAGE_TOTAL_TOKENS]: totalTokens,
        });
      }
    }

    // Also check for tokenUsage format (for compatibility)
    if (output.llmOutput?.tokenUsage) {
      const usage = output.llmOutput.tokenUsage;
      if (usage.promptTokens) {
        span.setAttributes({
          [ATTR_GEN_AI_USAGE_PROMPT_TOKENS]: usage.promptTokens,
        });
      }
      if (usage.completionTokens) {
        span.setAttributes({
          [ATTR_GEN_AI_USAGE_COMPLETION_TOKENS]: usage.completionTokens,
        });
      }
      if (usage.totalTokens) {
        span.setAttributes({
          [SpanAttributes.LLM_USAGE_TOTAL_TOKENS]: usage.totalTokens,
        });
      }
    }

    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
    this.spans.delete(runId);
  }

  async handleChatModelEnd(
    output: LLMResult,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _extraParams?: Record<string, unknown>,
  ): Promise<void> {
    // Same as handleLLMEnd for chat models
    return this.handleLLMEnd(output, runId, _parentRunId, _tags, _extraParams);
  }

  override async handleLLMError(
    err: Error,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _extraParams?: Record<string, unknown>,
  ): Promise<void> {
    const spanData = this.spans.get(runId);
    if (!spanData) return;

    const { span } = spanData;
    span.recordException(err);
    span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
    span.end();
    this.spans.delete(runId);
  }

  override async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    metadata?: Record<string, unknown>,
    runType?: string,
    runName?: string,
  ): Promise<void> {
    const startTime = Date.now();
    const chainName = chain.id?.[chain.id.length - 1] || "unknown";
    const spanName = `${chainName}.workflow`;

    const span = this.tracer.startSpan(spanName, {
      kind: SpanKind.CLIENT,
    });

    span.setAttributes({
      "traceloop.span.kind": "workflow",
      "traceloop.workflow.name": runName || chainName,
    });

    if (this.traceContent) {
      span.setAttributes({
        "traceloop.entity.input": JSON.stringify(inputs),
      });
    }

    this.spans.set(runId, { span, runId, startTime });
  }

  override async handleChainEnd(
    outputs: ChainValues,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _kwargs?: { inputs?: Record<string, unknown> },
  ): Promise<void> {
    const spanData = this.spans.get(runId);
    if (!spanData) return;

    const { span } = spanData;

    if (this.traceContent) {
      span.setAttributes({
        "traceloop.entity.output": JSON.stringify(outputs),
      });
    }

    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
    this.spans.delete(runId);
  }

  override async handleChainError(
    err: Error,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _kwargs?: { inputs?: Record<string, unknown> },
  ): Promise<void> {
    const spanData = this.spans.get(runId);
    if (!spanData) return;

    const { span } = spanData;
    span.recordException(err);
    span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
    span.end();
    this.spans.delete(runId);
  }

  override async handleToolStart(
    tool: Serialized,
    input: string,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _metadata?: Record<string, unknown>,
    _runName?: string,
  ): Promise<void> {
    const startTime = Date.now();
    const toolName = tool.id?.[tool.id.length - 1] || "unknown";
    const spanName = `${toolName}.task`;

    const span = this.tracer.startSpan(spanName, {
      kind: SpanKind.CLIENT,
    });

    span.setAttributes({
      "traceloop.span.kind": "task",
      "traceloop.entity.name": toolName,
    });

    if (this.traceContent) {
      span.setAttributes({
        "traceloop.entity.input": JSON.stringify({ args: [input] }),
      });
    }

    this.spans.set(runId, { span, runId, startTime });
  }

  override async handleToolEnd(
    output: any,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
  ): Promise<void> {
    const spanData = this.spans.get(runId);
    if (!spanData) return;

    const { span } = spanData;

    if (this.traceContent) {
      span.setAttributes({
        "traceloop.entity.output": JSON.stringify(output),
      });
    }

    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
    this.spans.delete(runId);
  }

  override async handleToolError(
    err: Error,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
  ): Promise<void> {
    const spanData = this.spans.get(runId);
    if (!spanData) return;

    const { span } = spanData;
    span.recordException(err);
    span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
    span.end();
    this.spans.delete(runId);
  }

  /**
   * Extract model name from the LLM configuration object (at start time)
   */
  private extractModelNameFromConfig(llm: Serialized): string | null {
    const llmAny = llm as unknown as Record<string, unknown>;
    const kwargs = llmAny.kwargs as Record<string, unknown> | undefined;

    // Check kwargs first (most common pattern in LangChain v1)
    if (kwargs) {
      // Azure deployment name (this is typically the model)
      if (
        kwargs.azureOpenAIApiDeploymentName &&
        typeof kwargs.azureOpenAIApiDeploymentName === "string"
      ) {
        return kwargs.azureOpenAIApiDeploymentName;
      }
      // Standard model name fields
      const modelName =
        kwargs.model_name ||
        kwargs.modelName ||
        kwargs.model ||
        kwargs.deployment_name ||
        kwargs.deploymentName;
      if (modelName && typeof modelName === "string") {
        return modelName;
      }
    }

    // Check direct properties on the llm object
    if (
      llmAny.azureOpenAIApiDeploymentName &&
      typeof llmAny.azureOpenAIApiDeploymentName === "string"
    ) {
      return llmAny.azureOpenAIApiDeploymentName;
    }
    const directModel =
      llmAny.model_name ||
      llmAny.modelName ||
      llmAny.model ||
      llmAny.deployment_name ||
      llmAny.deploymentName;
    if (directModel && typeof directModel === "string") {
      return directModel;
    }

    return null;
  }

  private extractModelNameFromResponse(output: LLMResult): string | null {
    // Check llmOutput first (multiple possible field names)
    if (output.llmOutput) {
      // Check various snake_case and camelCase variants
      const modelName =
        output.llmOutput.model_name ||
        output.llmOutput.modelName ||
        output.llmOutput.model_id ||
        output.llmOutput.modelId ||
        output.llmOutput.model;
      if (modelName && typeof modelName === "string") {
        return modelName;
      }
    }

    // Check generations for response_metadata (LangChain v1 pattern)
    if (output.generations && output.generations.length > 0) {
      const firstGen = output.generations[0];
      if (firstGen && firstGen.length > 0) {
        const generation = firstGen[0] as {
          message?: {
            response_metadata?: {
              model_name?: string;
              modelName?: string;
              model?: string;
            };
          };
          generationInfo?: {
            model_name?: string;
            modelName?: string;
            model?: string;
          };
        };

        // Check response_metadata on the message (LangChain v1 chat models)
        if (generation.message?.response_metadata) {
          const meta = generation.message.response_metadata;
          const model = meta.model_name || meta.modelName || meta.model;
          if (model && typeof model === "string") {
            return model;
          }
        }

        // Check generationInfo (older pattern)
        if (generation.generationInfo) {
          const info = generation.generationInfo;
          const model = info.model_name || info.modelName || info.model;
          if (model && typeof model === "string") {
            return model;
          }
        }
      }
    }

    return null;
  }

  private convertClassNameToSpanName(className: string): string {
    // Convert PascalCase to lowercase with dots
    // BedrockChat -> bedrock.chat
    // ChatOpenAI -> chat.openai
    return className.replace(/([A-Z])/g, (match, char, index) => {
      return index === 0 ? char.toLowerCase() : `.${char.toLowerCase()}`;
    });
  }

  /**
   * Detect the actual class name by checking multiple sources:
   * 1. Azure-specific properties (azureOpenAIApiKey, azureOpenAIApiDeploymentName, etc.)
   * 2. kwargs for Azure-specific properties
   * 3. Constructor name
   * 4. Fallback to serialized id
   */
  private detectClassName(llm: Serialized): string | null {
    // Type-safe access to llm properties
    const llmAny = llm as unknown as Record<string, unknown>;
    const kwargs = llmAny.kwargs as Record<string, unknown> | undefined;

    // Check _llmType() for Azure detection
    if (kwargs) {
      // Check for Azure-specific properties in kwargs
      if (
        kwargs.azureOpenAIApiKey ||
        kwargs.azureOpenAIApiDeploymentName ||
        kwargs.azureOpenAIApiInstanceName ||
        kwargs.azureOpenAIEndpoint ||
        kwargs.azureADTokenProvider
      ) {
        return "AzureChatOpenAI";
      }
    }

    // Check for Azure properties directly on the llm object
    // This handles cases where the llm is passed with configuration
    if (
      llmAny.azureOpenAIApiKey ||
      llmAny.azureOpenAIApiDeploymentName ||
      llmAny.azureOpenAIApiInstanceName ||
      llmAny.azureOpenAIEndpoint ||
      llmAny.azureADTokenProvider
    ) {
      return "AzureChatOpenAI";
    }

    // Check constructor name if available
    if (
      llmAny.constructor &&
      typeof llmAny.constructor === "function" &&
      (llmAny.constructor as { name?: string }).name
    ) {
      const constructorName = (llmAny.constructor as { name: string }).name;
      if (constructorName !== "Object" && constructorName !== "Function") {
        return constructorName;
      }
    }

    // Fallback to serialized id
    return llm.id?.[llm.id.length - 1] || null;
  }

  private detectVendor(llm: Serialized): string {
    // First, use the detected class name for more accurate vendor detection
    const detectedClassName = this.detectClassName(llm);
    const className = detectedClassName || llm.id?.[llm.id.length - 1] || "";

    // Type-safe access to llm properties
    const llmAny = llm as unknown as Record<string, unknown>;
    const kwargs = llmAny.kwargs as Record<string, unknown> | undefined;

    // Also check for Azure properties directly - this is the most reliable method
    // for LangChain v1 where wrapper classes are used
    if (kwargs) {
      if (
        kwargs.azureOpenAIApiKey ||
        kwargs.azureOpenAIApiDeploymentName ||
        kwargs.azureOpenAIApiInstanceName ||
        kwargs.azureOpenAIEndpoint ||
        kwargs.azureADTokenProvider
      ) {
        return "Azure";
      }
    }

    // Check Azure properties directly on the llm object
    if (
      llmAny.azureOpenAIApiKey ||
      llmAny.azureOpenAIApiDeploymentName ||
      llmAny.azureOpenAIApiInstanceName ||
      llmAny.azureOpenAIEndpoint ||
      llmAny.azureADTokenProvider
    ) {
      return "Azure";
    }

    if (!className) {
      return "Langchain";
    }

    // Follow Python implementation with exact matches and patterns
    // Ordered by specificity (most specific first)

    // Azure (most specific - check first)
    if (
      ["AzureChatOpenAI", "AzureOpenAI", "AzureOpenAIEmbeddings"].includes(
        className,
      ) ||
      className.toLowerCase().includes("azure")
    ) {
      return "Azure";
    }

    // OpenAI
    if (
      ["ChatOpenAI", "OpenAI", "OpenAIEmbeddings"].includes(className) ||
      className.toLowerCase().includes("openai")
    ) {
      return "openai";
    }

    // AWS Bedrock
    if (
      ["ChatBedrock", "BedrockEmbeddings", "Bedrock", "BedrockChat"].includes(
        className,
      ) ||
      className.toLowerCase().includes("bedrock") ||
      className.toLowerCase().includes("aws")
    ) {
      return "AWS";
    }

    // Anthropic
    if (
      ["ChatAnthropic", "AnthropicLLM"].includes(className) ||
      className.toLowerCase().includes("anthropic")
    ) {
      return "Anthropic";
    }

    // Google (Vertex/PaLM/Gemini)
    if (
      [
        "ChatVertexAI",
        "VertexAI",
        "VertexAIEmbeddings",
        "ChatGoogleGenerativeAI",
        "GoogleGenerativeAI",
        "GooglePaLM",
        "ChatGooglePaLM",
      ].includes(className) ||
      className.toLowerCase().includes("vertex") ||
      className.toLowerCase().includes("google") ||
      className.toLowerCase().includes("palm") ||
      className.toLowerCase().includes("gemini")
    ) {
      return "Google";
    }

    // Cohere
    if (
      ["ChatCohere", "CohereEmbeddings", "Cohere"].includes(className) ||
      className.toLowerCase().includes("cohere")
    ) {
      return "Cohere";
    }

    // HuggingFace
    if (
      [
        "HuggingFacePipeline",
        "HuggingFaceTextGenInference",
        "HuggingFaceEmbeddings",
        "ChatHuggingFace",
      ].includes(className) ||
      className.toLowerCase().includes("huggingface")
    ) {
      return "HuggingFace";
    }

    // Ollama
    if (
      ["ChatOllama", "OllamaEmbeddings", "Ollama"].includes(className) ||
      className.toLowerCase().includes("ollama")
    ) {
      return "Ollama";
    }

    // Together
    if (
      ["Together", "ChatTogether"].includes(className) ||
      className.toLowerCase().includes("together")
    ) {
      return "TogetherAI";
    }

    // Replicate
    if (
      ["Replicate", "ChatReplicate"].includes(className) ||
      className.toLowerCase().includes("replicate")
    ) {
      return "Replicate";
    }

    return "Langchain";
  }

  private mapMessageTypeToRole(messageType: string): string {
    // Map LangChain message types to standard OpenTelemetry roles
    switch (messageType) {
      case "human":
        return "user";
      case "ai":
        return "assistant";
      case "system":
        return "system";
      case "function":
        return "tool";
      default:
        return messageType;
    }
  }
}
