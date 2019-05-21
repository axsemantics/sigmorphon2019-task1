import json
import sys
from pathlib import Path

import click

def generate(language, meta):
    high_lang, low_lang = language.split("--")
    folder = Path(f"{high_lang}--{low_lang}")

    base_folder = "../conll2019/task1"
    data = json.load(open("configurations/sigmorphon2019-task1-system1.json"))

    train_all = base_folder / folder / "train-all"
    with open(train_all, "w") as fp:
        for _ in [f"{high_lang}-train-high", f"{low_lang}-train-low"]:
            with open(base_folder / folder / Path(_)) as fpr:
                for line in fpr.readlines():
                    fp.write(line)

    if meta == "transfer":
        data = json.load(open("configurations/sigmorphon2019-task1-system2.json"))
        fn = Path(f"system2_{language}.json")
        data["trainer"]["low_data_path"] = str(
            base_folder / folder / f"{low_lang}-train-low"
        )
    elif meta == "only_low":
        fn = Path(f"system1-low_{language}.json")
    else:
        fn = Path(f"system1_{language}.json")

    data["train_data_path"] = str(train_all)
    if meta == "only_low":
        data["train_data_path"] = str(base_folder / folder / f"{low_lang}-train-low")

    data["validation_data_path"] = str(base_folder / folder / f"{low_lang}-dev")

    json.dump(data, open(f"experiments.2019" / fn, "w"), indent=2)
    print(fn)


@click.command()
@click.option("--language", "-l")
@click.option("--meta", "-m")
def main(language, meta):
    generate(language, meta)


if __name__ == "__main__":
    main()


# example calls:
# python sigmorphon/gen_config.py -l adyghe--kabardian -m transfer
