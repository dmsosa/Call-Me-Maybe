import argparse
import sys

from src.core.tokenizer import Tokenizer
from src.exception.exception import InputError
from .storage.json_loader import FunctionStorage, \
	load_prompts, \
	load_fn_storage

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
	args = parse_args(argv)
	try:
		fn_storage = load_fn_storage(args.functions_definition)
		prompts = load_prompts(args.input)
		tokenizer = Tokenizer(fn_storage, prompts)
		tokenizer.pass_to_model()

	except InputError as e:
		print(f"Error: {e}", file=sys.stderr)
		return (1)
	return (0)

if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except KeyboardInterrupt as e:
		raise e
	except Exception as e:
		raise Exception(f"Unexpected exception raised while creating Function Storage: \n{e}\n")
