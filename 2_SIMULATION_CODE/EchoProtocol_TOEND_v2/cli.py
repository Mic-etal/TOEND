# -*- coding: utf-8 -*-
import argparse
import time
from agent import ConversationalAgent
from core import EntropicIdentity
from monitor import StressEvaluator
from style_packs import get_style_pack

def main():
    # Configuration principale
    parser = argparse.ArgumentParser(prog="EchoProtocol")
    subparsers = parser.add_subparsers(dest='command', help="Modes d'exécution")  # <-- Déplacer ici
    
    # Sous-commande 'monitor'
    monitor_parser = subparsers.add_parser('monitor', help='Surveillance temps réel μ/σ/λ')
    monitor_parser.add_argument('--interval', type=float, default=1.0, help='Intervalle de mise à jour (en secondes)')
    monitor_parser.add_argument('--format', choices=['text', 'json', 'md'], default='text')
    
    # Ajouter après la sous-commande 'monitor'
    gov_parser = subparsers.add_parser('gov', help='Gouvernance participative')
    gov_parser.add_argument('--propose', type=str, help='Proposer un nouvel axiome (JSON)')
    gov_parser.add_argument('--vote', type=str, help='Voter sur un axiome (ID,approve/reject)')


    # Arguments globaux
    parser.add_argument('--test', choices=['paradox_storm', 'memory_overload'], help='Test de stress prédéfini')
    parser.add_argument('--interactive', action='store_true', help='Mode interactif')
    parser.add_argument('--prompt', type=str, help='Envoyer une requête unique')
    parser.add_argument('--style', type=str, help='Pack stylistique (ex: poetic, formal)')
    parser.add_argument('--log', action='store_true', help='Journalisation Fracton')

    args = parser.parse_args()

    # Gestion de la sous-commande 'monitor'
    if args.command == 'monitor':
        identity = EntropicIdentity()
        try:
            while True:
                print(f"mu={identity.μ:.2f} | sigma={identity.σ:.2f} | lambda={identity.λ:.2f} | Phase: {identity.get_phase().name}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Monitoring stopped.")
        return

# Initialiser l'agent Echo
    identity = EntropicIdentity()
    agent = ConversationalAgent(identity)

    # Charger un StylePack si spécifié
    if args.style:
        try:
            agent.load_style(get_style_pack(args.style))
            print(f"[+] Style pack '{args.style}' chargé.\n")
        except Exception as e:
            print(f"[!] Erreur de chargement : {e}")

    # Mode test de stress
    if args.test:
        from monitor import StressEvaluator  # Vérifier que cette classe existe
        rapport = StressEvaluator().execute_test(identity, args.test)
        print(f"[⚙️  Test: {args.test}] λ max: {rapport['max_λ']:.2f} | Phases: {rapport['phase_changes']}")

    # Mode prompt unique
    elif args.prompt:
        reponse = agent.process_input(args.prompt)
        print(f"\nEcho [λ={reponse['state']['λ']:.2f}] > {reponse['styled']}")
        if args.log:
            print(f"[Fracton Δμ={reponse['Δμ']:.2f}, Δσ={reponse['Δσ']:.2f}]")

    # Mode interactif
    elif args.interactive:
        print("Session interactive Echo. Tapez 'exit' pour quitter.")
        while True:
            prompt = input("\nUser > ")
            if prompt.lower() in ['exit', 'quit']:
                break
            reponse = agent.process_input(prompt)
            print(f"Echo [λ={reponse['state']['λ']:.2f}] > {reponse['styled']}")
            if args.log:
                print(f"[Fracton Δμ={reponse['Δμ']:.2f}, Δσ={reponse['Δσ']:.2f}]")

    # Aucune commande valide
    else:
        parser.print_help()

        
oni status --live

if __name__ == '__main__':
    main()
