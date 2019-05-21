import csv
import json
from pathlib import Path

import click


@click.command()
@click.option("--affix", "-a")
@click.option("--language", "-l")
@click.option("--eval-path", "-ep")
def main(language, affix, eval_path=None):
    if not eval_path:
        eval_path = Path("eval.2019/")
    else:
        eval_path = Path(eval_path)

    high_lang, low_lang = language.split("--")
    folder = Path(f"{high_lang}--{low_lang}")

    base_folder = "../conll2019/task1/"
    with open(base_folder / folder / f"{low_lang}-{affix}") as fp_original, open(
            eval_path / f"{language}-{affix}.output"
    ) as fp_output, open(eval_path / f"{language}-{affix}", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")

        for line_output, line_original in zip(
                fp_output.readlines(), fp_original.readlines()
        ):
            original = line_original.strip().split("\t")
            output = json.loads(line_output)

            predicted_tokens = output["predicted_tokens"]
            if len(predicted_tokens) > 0 and isinstance(predicted_tokens[0], list):
                # beam_search, get best result
                predicted_tokens = predicted_tokens[0]

            infl = "".join(predicted_tokens)
            writer.writerow([original[0], infl, original[-1]])


if __name__ == "__main__":
    main()

# example calls:
# python sigmorphon/predict_to_sigmorphon.py -l adyghe--kabardian -a dev
