# generates charset and feature analysis and
# writes to a json file in eval.2019/analysis/

import csv
from pathlib import Path
import click
import json
from collections import Counter
from operator import itemgetter
from tqdm import tqdm


sigmorphon2019_path = Path("../conll2019/task1/")


def read_data(file_path):
    with open(file_path, "r") as data_file:
        for line_num, row in enumerate(csv.reader(data_file, delimiter="\t")):
            source_sequence, target_sequence, feature_sequence = row
            yield source_sequence, feature_sequence, target_sequence


def sort(d):
    return dict(sorted(d.items(), key=itemgetter(1), reverse=True))


def stats_by_pair(language):
    high, low = language.split("--")

    data = {}
    data["charset-high"] = Counter()
    data["charset-low"] = Counter()
    data["features-high"] = Counter()
    data["features-low"] = Counter()

    for t, names in (
        ("high", (f"{high}-train-high",)),
        ("low", (f"{low}-train-low", f"{low}-dev")),
    ):
        for name in names:
            for source, features, target in read_data(
                sigmorphon2019_path / language / name
            ):
                chars = Counter(source)
                data[f"charset-{t}"].update(chars)
                chars = Counter(target)
                data[f"charset-{t}"].update(chars)
                feats = Counter(features.split(";"))
                data[f"features-{t}"].update(feats)

        data[f"charset-{t}"] = sort(data[f"charset-{t}"])
        data[f"features-{t}"] = sort(data[f"features-{t}"])

        # charset analysis
        data[f"charset-{t}-percent"] = {}
        sum_chars = sum(data[f"charset-{t}"].values())
        for key in data[f"charset-{t}"].keys():
            value = data[f"charset-{t}"][key] / sum_chars
            data[f"charset-{t}-percent"][key] = f"{value:0.6f}"

        data[f"charset-{t}-percent"] = sort(data[f"charset-{t}-percent"])

    data["charset-high-not-low"] = list(
        set(data[f"charset-high"].keys()) - set(data[f"charset-low"].keys())
    )
    data["charset-low-not-high"] = list(
        set(data[f"charset-low"].keys()) - set(data[f"charset-high"].keys())
    )
    data["charset-stats"] = {
        "char-count-high": len(data[f"charset-high"]),
        "char-count-low": len(data[f"charset-low"]),
        "char-count-high-not-low": len(data[f"charset-high-not-low"]),
        "char-count-low-not-high": len(data[f"charset-low-not-high"]),
    }

    data["features-high-not-low"] = list(
        set(data[f"features-high"].keys()) - set(data[f"features-low"].keys())
    )
    data["features-low-not-high"] = list(
        set(data[f"features-low"].keys()) - set(data[f"features-high"].keys())
    )

    data["feature-stats"] = {
        "feature-count-high": len(data[f"features-high"]),
        "feature-count-low": len(data[f"features-low"]),
        "feature-count-high-not-low": len(data[f"features-high-not-low"]),
        "feature-count-low-not-high": len(data[f"features-low-not-high"]),
    }

    path = Path("eval.2019/analysis")
    path.mkdir(exist_ok=True, parents=True)
    json.dump(data, open(path / f"{language}.json", "w"), indent=2, ensure_ascii=False)


def gen_table():
    path = Path("eval.2019/analysis")
    with open(path / 'analysis.csv', 'w') as csvfile:
        fieldnames = ['language_pair', 'feature-count-low-not-high', 'char-count-low-not-high']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for fn in sorted(path.glob("*--*")):
            data = json.load(open(fn))
            writer.writerow({
                'language_pair': fn.name.split('.')[0],
                'feature-count-low-not-high': data["feature-stats"]["feature-count-low-not-high"],
                'char-count-low-not-high': data["charset-stats"]["char-count-low-not-high"]
            })


@click.command()
@click.option("--language", "-l", required=False)
@click.option("--table", "-t", is_flag=True, default=False)
def main(language, table):
    if table:
        gen_table()
        return

    if language:
        languages = [language]
    else:
        languages = [i.name for i in sigmorphon2019_path.glob("*--*")]

    pbar = tqdm(sorted(languages), unit="pairs")
    for language in pbar:
        pbar.set_description(f"processing: {language:>30}")
        stats_by_pair(language)


if __name__ == "__main__":
    main()


# example calls:
# python sigmorphon/feature_analysis.py
# python sigmorphon/feature_analysis.py -l adyghe--kabardian
