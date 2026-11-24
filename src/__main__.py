import argparse
from pathlib import Path
import sys
import time
import traceback
from src.core.utils import fmt_time
from src.exception.exception import InputError, VisualizerQuit
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
        description="Constrained function calling for the Call Me Maybe project. "
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
        help="Path where the metrics are going to be saved.",
    )
    parser.add_argument(
        "--cache_results",
        action="store_true",
        help="Store output results of function calls in a cache file.",
        default=False
    )
    parser.add_argument(
        "--cache_enabled",
        action="store_true",
        help="Load output results of function calls from the cache file.",
        default=False
    )
    parser.add_argument(
        "--model",
        help="Specify the model to use.",
        default="Qwen/Qwen3-0.6B"
    )
    parser.add_argument(
        "--visual",
        help="Activate a visualization Graphical User Interface",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--step",
        help="Visualization step by step",
        default=False,
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
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            print(
                "\nLoading LLM Model,"
                " the first time this may take a while..."
                )

        start = time.time()

        from llm_sdk import Small_LLM_Model
        from src.core.llm import LargeLanguageModel
        from src.core.generator import Generator

        model = Small_LLM_Model(model_name=args.model)
        llm = LargeLanguageModel(model=model)

    except Exception as e:
        print(
            f"Error: could not load the model: {e}",
            file=sys.stderr
            )
        return 1

    print(f"    model loaded in {fmt_time(time.time() - start)}",
          file=sys.stderr)

    model_load_s = time.time() - start
    on_change = None
    visualizer = None
    if args.visual and sys.stdout.isatty():
        visualizer = CmmGui(
            model_name=args.model,
            step_mode=args.step,
            started_at=start,
            )
        on_change = visualizer.on_change

    # In this phase, once the model is loaded,
    # Start generating JSON for each prompt
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
            on_change=on_change,
            cache_enabled=args.cache_enabled
        )
    try:
        lang_generator.run()
    except VisualizerQuit:
        print("Visualization interrupted by the user")
    finally:
        if visualizer is not None:
            visualizer.close()          # sort de l'ecran alterne + resume

    print(f"\n{'='*100}")
    print(
        "   Completed: "
        f"{len(lang_generator.results) - lang_generator.errors_count} "
        " successes, "
        f"{lang_generator.errors_count} "
        "errors"
        )
    print(f"   Results in: {args.output}")
    print(f"   done in {fmt_time(lang_generator.metrics.finished_at - lang_generator.metrics.started_at)} -> {args.output}",
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
        traceback.print_exc()
        raise (SystemExit(1))
