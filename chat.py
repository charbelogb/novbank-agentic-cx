import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from main import get_customer_transactions
from datetime import date
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

load_dotenv(Path(__file__).parent / ".env")

# Identité fournie par l'application pour notre démonstration.
CURRENT_CUSTOMER_ID = "CUST-001"


@tool
def list_my_transactions() -> list[dict]:
    """Consulte les transactions du client actuellement identifié.

    À utiliser pour rechercher une opération mentionnée par le client :
    retrait au distributeur, paiement par carte ou autre transaction.
    """
    return get_customer_transactions(CURRENT_CUSTOMER_ID)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY est absente du fichier .env.")

    model = ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        reasoning_effort="none",
    )

    model_with_tools = model.bind_tools([list_my_transactions])

    messages: list[BaseMessage] = [
        SystemMessage(
            content = f"""
            Tu es l'assistant de service client de NovBank.
            Réponds en français, de manière claire et concise.

            Le client est déjà identifié par l'application.
            Ne lui demande pas ses identifiants ou son numéro de compte.
            
            Tu disposes de l'outil list_my_transactions pour consulter ses transactions.
            Lorsqu'un client signale un problème sur une opération,
            consulte ses transactions avant de te prononcer sur cette opération.
            
            N'invente aucune transaction, aucune action ou aucun délai.
            Ne promets pas de remboursement.
            
            La date du jour est le {date.today().isoformat()}.
            
            Après consultation, recherche la transaction correspondant
            au type d'opération, au montant et à la date mentionnés.

            Si une transaction correspond, présente-la et demande confirmation.
            Si plusieurs correspondent, demande une précision pour les départager.
            Si aucune ne correspond, indique-le et demande de vérifier les détails.

            Le statut DEBITED confirme le débit enregistré, mais ne prouve pas
            que le distributeur a remis les billets.

            Tu peux uniquement consulter les transactions.
            Tu ne peux pas encore créer de réclamation ou effectuer de remboursement.
            """,
        ),
        HumanMessage(
            content="J’ai retiré 100 000 FCFA hier. Mon compte a été débité "
                "mais le distributeur ne m’a rien donné.",
        ),
    ]

    # 1. Le modèle reçoit la demande et peut demander un outil.
    response = model_with_tools.invoke(messages)

    # On conserve sa demande dans l'historique.
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Notre programme contrôle les outils autorisés.
            if tool_call["name"] != "list_my_transactions":
                raise ValueError(f"Outil inconnu : {tool_call['name']}")

            print(f"\nExécution de l'outil : {tool_call['name']}")

            # 2. Python exécute l'outil, qui interroge SQLite.
            tool_message = list_my_transactions.invoke(tool_call)

            print("Résultat de l'outil :")
            print(tool_message.content)

            # Le résultat rejoint l'historique transmis au modèle.
            messages.append(tool_message)

        # 3. Le modèle formule sa réponse à partir des résultats.
        final_response = model.invoke(messages)
    else:
        # Le modèle a répondu directement, sans demander d'outil.
        final_response = response

    print("\nRéponse de NovBank :")
    print(final_response.content)

if __name__ == "__main__":
    main()
