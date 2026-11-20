import argparse
from pathlib import Path
import sys
import time
from src.core.utils import fmt_time
from src.exception.exception import InputError
from src.model.metrics import CmmMetrics
from src.storage.json_writer import JsonWriter
from src.visual.gui import CmmGui
from .storage.json_loader import load_prompts, \
    load_function_list
import os


LLM_SDK_AVAILABLE = True


def init_argparse() -> argparse.Namespace:
    """Parser of the terminal code """
    parser = argparse.ArgumentParser(
        description="Constrained function calling (Marco 7)."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    parser.add_argument(
        "--cache_model", "-c",
        action="store_true",
        help="Load model from local cache instead of downloading.",
        default=False
    )
    parser.add_argument(
        "--metrics_path",
        default="data/output/metrics.json",
    )
    parser.add_argument(
        "--cache_results",
        action="store_true",
        help="Store output results of function calls in a cache file.",
        default="False"
    )
    parser.add_argument(
        "--model",
        help="Specify the model to use.",
        default="Qwen/Qwen3-0.6B"
    )
    parser.add_argument(
        "--visualize",
        help="Activate a visualization Graphical User Interface",
        default="False",
        action="store_true"
    )

    return parser.parse_args()


def main() -> int:
    """
    After storing the JSON, we need to start a loop through
    each prompt to start generating the output function calls.
    """
    args = init_argparse()
    print("Loading input files...")
    try:
        fn_list = load_function_list(args.functions_definition)
        prompt_list = load_prompts(args.input)
    except InputError as e:
        print(f"Error: {e}", file=sys.stderr)
        return (1)

    fn_count = len(fn_list.root)
    prompts_count = len(prompt_list.root)
    if fn_count == 0:
        print(
            "Error: Zero (0) functions loaded, "
            "can not continue with the program",
            file=sys.stderr
            )
        return (1)
    if prompts_count == 0:
        print(
            "Error: Zero (0) prompts loaded, "
            "can not continue with the program",
            file=sys.stderr
            )
        return (1)

    print(f"    {fn_count} functions loaded successfully...")
    print(f"    {prompts_count} prompts loaded successfully...")

    try:
        if args.cache_model:
            print("\nLoading LLM Model from cache...")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        else:
            print(
                "\nLoading LLM Model,"
                "the first time this may take a while..."
                )

        start = time.time()

        from llm_sdk import Small_LLM_Model
        from src.core.llm import LargeLanguageModel
        from src.core.generator import Generator

        model = Small_LLM_Model(model_name=args.model)
        llm = LargeLanguageModel(model=model)

    except Exception as e:
        print(
            f"Error: could not load the model: {e}\n"
            "Suggestion: The model Qwen/Qwen3-0.6B is not"
            "loaded in the HF local cache, try running the"
            "first time with --load-model or -l to load the model",
            file=sys.stderr
            )
        return 1

    print(f"    model loaded in {fmt_time(time.time() - start)}",
          file=sys.stderr)

    model_load_s = time.time() - start
    # In this phase, once the model is loaded,
    # Start generating JSON for each prompt
    try:
        lang_generator = Generator(
                llm=llm,
                fn_list=fn_list.root,
                prompts=prompt_list.root,
                writer=JsonWriter(output_path=Path(args.output)),
                metrics=CmmMetrics(
                    metrics_path=Path(args.metrics_path),
                    started_at=start,
                    model_load_s=model_load_s
                    ),
            )
        on_change = None
        if args.visualize:
            if sys.stdout.isatty():
                visualizer = CmmGui(
                    model_name=args.model,
                    start=start
                )
                on_change = visualizer.on_change
            else:
                print("Can not activate visualzier: program is not running in a terminal")
            
    lang_generator.run()

    print(f"\n{'='*50}")
    print(
        f"Completed: \
        {len(lang_generator.results) - lang_generator.errors} \
        successes, \
        {lang_generator.errors} \
        errors"
        )
    print(f"Results in: {args.output}")
    print(f"done in {fmt_time(time.time() - start)} -> {args.output}",
          file=sys.stderr)

    return (0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "user pressed Ctrl + C (KeyboardInterrupt)",
            file=sys.stderr
            )
        raise (SystemExit(1))
    except Exception as e:
        print(
            "Handled exception while "
            f"running main function \n{e}\n",
            file=sys.stderr
            )
        raise (SystemExit(1))
