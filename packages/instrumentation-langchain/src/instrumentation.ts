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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { context } from "@opentelemetry/api";
import {
  InstrumentationBase,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { version } from "../package.json";
import { TraceloopCallbackHandler } from "./callback_handler";
import { LangChainInstrumentationConfig } from "./types";

export class LangChainInstrumentation extends InstrumentationBase {
  declare protected _config: LangChainInstrumentationConfig;

  constructor(config: LangChainInstrumentationConfig = {}) {
    super("@traceloop/instrumentation-langchain", version, config);
  }

  public override setConfig(config: LangChainInstrumentationConfig = {}) {
    super.setConfig(config);
  }

  protected init(): InstrumentationModuleDefinition[] {
    this._diag.debug("[Traceloop] init() called - setting up module hooks");

    // List of all LangChain provider modules to patch
    const modulesToPatch = [
      "@langchain/openai",
      "@langchain/anthropic",
      "@langchain/google-genai",
      "@langchain/mistralai",
      "@langchain/aws",
      "@langchain/community",
    ];

    const llmModuleDefinitions = modulesToPatch.map((moduleName) =>
      this.createModuleDefinition(moduleName),
    );

    // Also patch @langchain/core/callbacks/manager to intercept CallbackManager
    const callbackManagerModuleDefinition =
      this.createCallbackManagerModuleDefinition();

    return [...llmModuleDefinitions, callbackManagerModuleDefinition];
  }

  /**
   * Create an InstrumentationModuleDefinition for @langchain/core/callbacks/manager
   * to patch CallbackManager and ensure our callback handler is always added
   */
  private createCallbackManagerModuleDefinition(): InstrumentationNodeModuleDefinition {
    return new InstrumentationNodeModuleDefinition(
      "@langchain/core/callbacks/manager",
      [">=0.1.0 <2.0.0"],
      (moduleExports: Record<string, unknown>, moduleVersion?: string) => {
        this._diag.debug(
          `[Traceloop] Applying CallbackManager instrumentation patch for @langchain/core/callbacks/manager@${moduleVersion || "unknown"}`,
        );
        this.patchCallbackManagerModule(moduleExports);
        return moduleExports;
      },
      (moduleExports: Record<string, unknown>, moduleVersion?: string) => {
        this._diag.debug(
          `[Traceloop] Removing CallbackManager instrumentation patch for @langchain/core/callbacks/manager@${moduleVersion || "unknown"}`,
        );
        return moduleExports;
      },
    );
  }

  /**
   * Patch @langchain/core/callbacks/manager module - specifically the CallbackManager
   */
  private patchCallbackManagerModule(module: Record<string, unknown>) {
    if (!module) return;

    // CallbackManager is directly exported from this module
    const callbacksManager = module.CallbackManager;

    if (callbacksManager) {
      this.patchCallbackManager(callbacksManager);
    } else {
      this._diag.debug(
        "[Traceloop] CallbackManager not found in @langchain/core/callbacks/manager",
      );
    }
  }

  /**
   * Patch CallbackManager.configure to always include our callback handler
   */
  private patchCallbackManager(CallbackManagerClass: unknown) {
    if (!CallbackManagerClass) return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const CM = CallbackManagerClass as any;

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;

    // Patch configure method
    if (CM.configure) {
      const originalConfigure = CM.configure;
      CM.configure = async function (...args: unknown[]) {
        self._diag.debug("[Traceloop] CallbackManager.configure called");

        const callbackHandler = new TraceloopCallbackHandler(
          self.tracer,
          self._shouldSendPrompts(),
        );

        // args[1] is localHandlers in the configure signature
        const localHandlers = args[1] as unknown[] | undefined;
        const patchedLocalHandlers = localHandlers
          ? [...localHandlers, callbackHandler]
          : [callbackHandler];

        // Create new args array with patched localHandlers
        const patchedArgs = [...args];
        patchedArgs[1] = patchedLocalHandlers;

        return originalConfigure.apply(this, patchedArgs);
      };
      self._diag.debug(
        "[Traceloop] CallbackManager.configure patched successfully",
      );
    }
  }

  /**
   * Create an InstrumentationModuleDefinition for a LangChain provider module.
   * This hooks into the module loading process to patch the correct module instance.
   */
  private createModuleDefinition(
    moduleName: string,
  ): InstrumentationNodeModuleDefinition {
    return new InstrumentationNodeModuleDefinition(
      moduleName,
      [">=0.1.0 <2.0.0"], // Support versions 0.1.x through 1.x
      (moduleExports: Record<string, unknown>, moduleVersion?: string) => {
        this._diag.debug(
          `[Traceloop] Applying instrumentation patch for ${moduleName}@${moduleVersion || "unknown"}`,
        );
        this.patchLLMModule(
          moduleExports,
          `${moduleName}@${moduleVersion || "unknown"}`,
        );
        return moduleExports;
      },
      (moduleExports: Record<string, unknown>, moduleVersion?: string) => {
        this._diag.debug(
          `[Traceloop] Removing instrumentation patch for ${moduleName}@${moduleVersion || "unknown"}`,
        );
        // Unpatch logic could be added here if needed
        return moduleExports;
      },
    );
  }

  private patchLLMModule(module: Record<string, unknown>, source: string) {
    if (!module) return;

    // Patch known LLM classes
    const llmClasses = [
      "AzureChatOpenAI",
      "ChatOpenAI",
      "AzureOpenAI",
      "OpenAI",
      "ChatAnthropic",
      "ChatGoogleGenerativeAI",
      "ChatMistralAI",
      "ChatBedrockConverse",
      "ChatBedrock",
    ];

    for (const className of llmClasses) {
      if (module[className]) {
        this.patchLLMClass(
          module[className] as new (...args: unknown[]) => unknown,
          className,
          source,
        );
      }
    }
  }

  private patchLLMClass(
    LLMClass: new (...args: unknown[]) => unknown,
    className: string,
    source: string,
  ) {
    if (!LLMClass || !LLMClass.prototype) return;

    const proto = LLMClass.prototype as Record<string, unknown>;
    if (proto._traceloopPatched) {
      this._diag.debug(`[Traceloop] ${className} (${source}) already patched`);
      return;
    }

    this._diag.debug(`[Traceloop] Patching ${className} (${source})`);

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;

    // Patch invoke method
    if (proto.invoke) {
      const originalInvoke = proto.invoke as (
        ...args: unknown[]
      ) => Promise<unknown>;
      proto.invoke = function (
        input: unknown,
        options?: Record<string, unknown>,
      ) {
        self._diag.debug(`[Traceloop] ${className}.invoke called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalInvoke.call(this, input, patchedOptions);
      };
    }

    // Patch stream method
    if (proto.stream) {
      const originalStream = proto.stream as (
        ...args: unknown[]
      ) => AsyncIterable<unknown>;
      proto.stream = function (
        input: unknown,
        options?: Record<string, unknown>,
      ) {
        self._diag.debug(`[Traceloop] ${className}.stream called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalStream.call(this, input, patchedOptions);
      };
    }

    // Patch generate method (also used directly by some code)
    if (proto.generate) {
      const originalGenerate = proto.generate as (
        ...args: unknown[]
      ) => Promise<unknown>;
      proto.generate = function (
        messages: unknown,
        options?: Record<string, unknown>,
        callbacks?: unknown[],
      ) {
        self._diag.debug(`[Traceloop] ${className}.generate called`);
        // generate has a different signature - callbacks can be passed as 3rd arg
        let opts = options || {};
        if (callbacks && !opts.callbacks) {
          opts = { ...opts, callbacks };
        }
        const patchedOptions = self.injectCallbackHandler(opts, className);
        return originalGenerate.call(this, messages, patchedOptions);
      };
    }

    // Patch _generate method (internal method called by chains)
    if (proto._generate) {
      const originalInternalGenerate = proto._generate as (
        ...args: unknown[]
      ) => Promise<unknown>;
      proto._generate = function (
        messages: unknown,
        options?: Record<string, unknown>,
        runManager?: unknown,
      ) {
        self._diag.debug(`[Traceloop] ${className}._generate called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalInternalGenerate.call(
          this,
          messages,
          patchedOptions,
          runManager,
        );
      };
    }

    // Patch _call method (used internally by chains via .pipe())
    if (proto._call) {
      const originalCall = proto._call as (
        ...args: unknown[]
      ) => Promise<unknown>;
      proto._call = function (
        messages: unknown,
        options?: Record<string, unknown>,
        runManager?: unknown,
      ) {
        self._diag.debug(`[Traceloop] ${className}._call called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalCall.call(this, messages, patchedOptions, runManager);
      };
    }

    // Patch _transform method (used when LLM is in a chain/pipe for streaming)
    if (proto._transform) {
      const originalTransform = proto._transform as (
        ...args: unknown[]
      ) => AsyncIterable<unknown>;
      proto._transform = function (
        generator: unknown,
        runManager?: unknown,
        options?: Record<string, unknown>,
      ) {
        self._diag.debug(`[Traceloop] ${className}._transform called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalTransform.call(
          this,
          generator,
          runManager,
          patchedOptions,
        );
      };
    }

    // Patch _streamIterator method (used by chains when streaming via RunnableSequence)
    if (proto._streamIterator) {
      const originalStreamIterator = proto._streamIterator as (
        ...args: unknown[]
      ) => AsyncIterable<unknown>;
      proto._streamIterator = function (
        input: unknown,
        options?: Record<string, unknown>,
      ) {
        self._diag.debug(`[Traceloop] ${className}._streamIterator called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalStreamIterator.call(this, input, patchedOptions);
      };
    }

    // Patch batch method
    if (proto.batch) {
      const originalBatch = proto.batch as (
        ...args: unknown[]
      ) => Promise<unknown>;
      proto.batch = function (
        inputs: unknown[],
        options?: Record<string, unknown>,
        batchOptions?: unknown,
      ) {
        self._diag.debug(`[Traceloop] ${className}.batch called`);
        const patchedOptions = self.injectCallbackHandler(options, className);
        return originalBatch.call(this, inputs, patchedOptions, batchOptions);
      };
    }

    proto._traceloopPatched = true;
    this._diag.debug(
      `[Traceloop] ${className} (${source}) patched successfully`,
    );
  }

  /**
   * Inject the TraceloopCallbackHandler into the options.callbacks array
   */
  private injectCallbackHandler(
    options: Record<string, unknown> | undefined,
    className: string,
  ): Record<string, unknown> {
    const callbackHandler = new TraceloopCallbackHandler(
      this.tracer,
      this._shouldSendPrompts(),
    );

    if (!options) {
      return { callbacks: [callbackHandler] };
    }

    // Clone options to avoid mutating the original
    const patchedOptions = { ...options };

    if (!patchedOptions.callbacks) {
      patchedOptions.callbacks = [callbackHandler];
    } else if (Array.isArray(patchedOptions.callbacks)) {
      // Check if TraceloopCallbackHandler is already in the array
      const hasTraceloop = (
        patchedOptions.callbacks as Array<{ name?: string }>
      ).some((cb) => cb && cb.name === "traceloop_callback_handler");
      if (!hasTraceloop) {
        patchedOptions.callbacks = [
          ...(patchedOptions.callbacks as unknown[]),
          callbackHandler,
        ];
      }
    } else {
      // callbacks is a CallbackManager - add our handler
      const callbackManager = patchedOptions.callbacks as {
        addHandler?: (handler: unknown) => void;
      };
      if (callbackManager.addHandler) {
        callbackManager.addHandler(callbackHandler);
      }
    }

    this._diag.debug(
      `[Traceloop] Injected callback handler for ${className}, callbacks count:`,
      Array.isArray(patchedOptions.callbacks)
        ? patchedOptions.callbacks.length
        : "CallbackManager",
    );

    return patchedOptions;
  }

  private _shouldSendPrompts() {
    const contextShouldSendPrompts = context
      .active()
      .getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT);

    if (contextShouldSendPrompts !== undefined) {
      return !!contextShouldSendPrompts;
    }

    return this._config.traceContent !== undefined
      ? this._config.traceContent
      : true;
  }
}
