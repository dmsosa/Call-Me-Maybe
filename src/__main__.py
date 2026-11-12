import argparse
from pathlib import Path
import sys
import time

from src.exception.exception import InputError
from src.storage.json_writer import JsonWriter
from .storage.json_loader import load_prompts, \
    load_fn_storage

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
    try:
        fn_storage = load_fn_storage(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as e:
        print(f"Error: {e}", file=sys.stderr)
        return (1)


    try:
        start = time.time()
 
        from llm_sdk import Small_LLM_Model
        from src.core.tokenizer import Generator

        gen = Generator(
            fn_storage=fn_storage,
            prompts=prompts,
            writer=JsonWriter(output_path=Path(args.output))
        )
        model = Small_LLM_Model()
    
    except Exception as e:
        print(f"Error: could not load the model: {e}", file=sys.stderr)
        return 1
    print(f"model ready in {fmt_time(time.time() - start)}",
          file=sys.stderr)
    # In this phase, once the model is loaded,
    # Start generating JSON for each prompt
    for prompt in prompts.root:
        json = gen.generate_json(model, prompt.prompt)
        print(json)
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