import datasets
import yaml
from tqdm import tqdm


def main() -> None:
    dataset_path = "alexandrainst/m_mmlu"

    # Removed hy and sk subdataset because the original dataset is broken
    # I created this PR https://huggingface.co/datasets/alexandrainst/m_mmlu/discussions/3
    # on the dataset for the authors, in case it will be accepeted the filter can be removed
    keys_without_hy_sk = list(  # noqa: F841
        filter(
            lambda k: ("hy" not in k and "sk" not in k),
            datasets.get_dataset_infos(dataset_path).keys(),
        )
    )

    for task in tqdm():
        file_name = f"m_mmlu_{task}.yaml"
        try:
            with open(f"{file_name}", "w") as f:
                f.write("# Generated by _generate_configs.py\n")
                yaml.dump(
                    {
                        "include": "_default_yaml",
                        "task": f"{dataset_path.split('/')[-1]}_{task}",
                        "dataset_name": task,
                    },
                    f,
                )
        except FileExistsError:
            pass


if __name__ == "__main__":
    main()
