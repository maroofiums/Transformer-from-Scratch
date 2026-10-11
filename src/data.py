from datasets import load_dataset

def load_translation_dataset() -> datasets.DatasetDict:
    """
    Load the translation dataset.
    """
    dataset = load_dataset(
        "Helsinki-NLP/opus_books",
        "en-fr"
    )
    return dataset  


def inspect_dataset(
    dataset: datasets.DatasetDict,
    num_examples: int = 5
) -> None:

    """
    Inspect the dataset by printing a few examples.
    """
    print(f"Available splits: {list(dataset.keys())}")
    
    train_data = dataset["train"]
    
    print("Training examples: ", len(train_data))
    print("\nSample sentence pairs:\n")

    for index in range(min(num_examples, len(train_data))):
        
        example = train_data[index]
        translation = example["translation"]

        print(f"Example {index + 1}:")
        print(f"  English: {translation['en']}")
        print(f"  French: {translation['fr']}\n")
        print("-" * 50)


if __name__ == "__main__":
    dataset = load_translation_dataset()
    inspect_dataset(dataset)