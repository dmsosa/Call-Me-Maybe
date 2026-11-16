import argparse
from pathlib import Path
import sys
import time

from src.core.utils import fmt_time
from src.exception.exception import InputError
from src.storage.json_writer import JsonWriter
from .storage.json_loader import load_prompts, \
    load_function_list

LLM_SDK_AVAILABLE = True

def init_argparse(argv: list[str] | None = None) -> argparse.Namespace:
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

    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    """
    Jetzt, nachdem ich alle meinen JSON Dateien gespeichert habe,
    sollte ich ein while loop starten, das nich beendent, bis das
    JSON Output bereit ist.

    Kann ich ein program erstellen, der JSON Ladst, und anderer JSON
    als Ergebniss gibt?
    """
    args = init_argparse(argv)
    print("Loading input files...")
    try:
        fn_list = load_function_list(args.functions_definition)
        prompt_list = load_prompts(args.input)
    except InputError as e:
        print(f"Error: {e}", file=sys.stderr)
        return (1)

    print(f"    {len(fn_list.root)} functions loaded successfully...")
    print(f"    {len(prompt_list.root)} prompts loaded successfully...")

    try:
        print(f"\nLoading LLM Model...")
        start = time.time()
 
        from llm_sdk import Small_LLM_Model
        from src.core.llm import LargeLanguageModel
        from src.core.generator import Generator

        model = Small_LLM_Model()
        llm = LargeLanguageModel(model=model)

    except Exception as e:
        print(f"Error: could not load the model: {e}", file=sys.stderr)
        return 1
    
    print(f"    model loaded in {fmt_time(time.time() - start)}",
          file=sys.stderr)
    
    # In this phase, once the model is loaded,
    # Start generating JSON for each prompt
    lang_generator = Generator(
            llm=llm,
            fn_list=fn_list.root,
            prompts=prompt_list.root,
            writer=JsonWriter(output_path=Path(args.output))
        )
    lang_generator.run()
    print(f"\n{'='*50}")
    print(f"Completed: {len(lang_generator.results)} successes, {lang_generator.errors} errors")
    print(f"Results in: {args.output}")
    print(f"done in {fmt_time(time.time() - start)} -> {args.output}",
          file=sys.stderr)


    return (0)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as e:
        print(
            f"user pressed Ctrl + C (KeyboardInterrupt)"
            , file=sys.stderr
            )
        raise(SystemExit(1))
    except Exception as e:
        print(
            f"Unexpected exception raised while running main function \n{e}\n"
            , file=sys.stderr)
        raise(SystemExit(1))