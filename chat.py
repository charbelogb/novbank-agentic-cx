import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from main import get_customer_transactions

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

def build_graph(model: ChatOpenAI):
    # Autorise le modèle à demander l'exécution de cet outil.
    model_with_tools = model.bind_tools([list_my_transactions])

    def call_model(state: MessagesState):
        # Envoie tout l'historique au modèle et ajoute sa réponse au graphe.
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def execute_tools(state: MessagesState):
        # Seul le dernier message peut contenir les demandes d'outils à exécuter.
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            raise TypeError("Un message du modèle était attendu.")

        results = []

        for tool_call in last_message.tool_calls:
            # Refuse toute demande d'outil qui n'est pas autorisée.
            if tool_call["name"] != "list_my_transactions":
                raise ValueError(
                    f"Outil inconnu : {tool_call['name']}"
                )

            print("\n[Consultation des transactions...]")

            # Exécute l'outil demandé et conserve son résultat dans l'historique.
            tool_message = list_my_transactions.invoke(tool_call)
            results.append(tool_message)

        return {"messages": results}

    def choose_next_step(
            state: MessagesState,
    ) -> Literal["tools", "finish"]:
        # Après la réponse du modèle, exécute ses outils s'il en a demandé;
        # sinon, le graphe peut se terminer.
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "finish"

    # Définit les deux étapes du graphe : réponse du modèle et exécution d'outils.
    builder = StateGraph(MessagesState)

    builder.add_node("assistant", call_model)
    builder.add_node("tools", execute_tools)

    # Commence par le modèle, puis choisit entre l'exécution d'outils et la fin.
    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        choose_next_step,
        {
            "tools": "tools",
            "finish": END,
        },
    )

    # Après un outil, retourne au modèle pour qu'il réponde avec son résultat.
    builder.add_edge("tools", "assistant")

    # Compile la définition en graphe exécutable.
    return builder.compile()

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY est absente du fichier .env.")

    model = ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        reasoning_effort="none",
    )

    graph = build_graph(model)

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
        )
    ]

    print("NovBank — Bonjour, comment puis-je vous aider ?")
    print("Tapez 'quitter' pour terminer.\n")

    while True:
        user_input = input("Vous : ").strip()

        if user_input.lower() in {"quitter", "exit"}:
            print("NovBank — À bientôt.")
            break

        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        result = graph.invoke(
            {"messages": messages},
            config={"recursion_limit": 10},
        )

        # Récupération de l'historique enrichi par le graphe.
        messages = result["messages"]

        print(f"\nNovBank : {messages[-1].content}\n")

if __name__ == "__main__":
    main()
