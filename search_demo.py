from knowledge import get_retriever


def main() -> None:
    retriever = get_retriever()

    question = (
        "J'ai fait un retrait au distributeur. "
        "Mon compte a été débité mais aucun billet n'est sorti. "
        "Que faut-il faire ?"
    )

    results = retriever.retrieve(question)

    print(f"Question : {question}")
    print(f"Passages retrouvés : {len(results)}")

    for position, result in enumerate(results, start=1):
        score = (
            f"{result.score:.3f}"
            if result.score is not None
            else "non disponible"
        )

        print(f"\n--- Passage {position} ---")
        print(f"Source : {result.node.metadata.get('source')}")
        print(f"Score : {score}")
        print(result.node.get_content())


if __name__ == "__main__":
    main()