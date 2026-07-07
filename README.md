# AgathaAI - Version Portfolio

AgathaAI est une copie anonymisée et non exécutable d'un projet local d'assistant IA / VTuber. Ce dossier sert uniquement de démonstration de code pour portfolio : il montre l'architecture, l'orchestration, les outils de fine-tuning, les intégrations temps réel et les choix techniques, sans fournir les données ou les assets nécessaires pour lancer le projet.

Tous droits réservés. Le code, l'architecture, les scripts, les idées d'intégration et les éléments de ce portfolio appartiennent à Axel. Aucune licence d'utilisation, de copie, de modification, de redistribution, d'entraînement, de déploiement ou d'exploitation commerciale n'est accordée. Seul Axel est autorisé à réutiliser, modifier, publier ou exploiter ce projet.

## Important

Ce dossier n'est pas une release et ne doit pas être exécuté. Il manque volontairement les modèles, datasets, fichiers de contexte, voix, secrets, tokens, fichiers `.env`, logs, exports, checkpoints et données runtime. Certains chemins ou imports pointent donc vers des éléments absents : c'est normal.

L'objectif est de permettre une lecture du code et une évaluation des compétences techniques, pas de fournir une application installable.

## Ce Que Fait Le Projet

AgathaAI est un runtime modulaire pour personnage IA. Il peut recevoir des messages depuis Discord, Twitch, une entrée vocale, un panneau web de contrôle et des intégrations de jeu. Ces entrées sont routées vers un modèle de langage local, une mémoire RAG, des outils natifs, une synthèse vocale, un système de modération et des hooks VTuber.

En version simple : c'est un système de personnage IA capable d'écouter, répondre, parler, se souvenir, modérer un chat, réagir à des événements de stream et exposer son état dans une interface web.

## Ce Que Cette Version Montre

- Orchestration asynchrone Python pour plusieurs services temps réel.
- Intégration Discord texte/voix avec un bot Node.js séparé.
- Connexion à un serveur LLM local compatible avec l'API OpenAI via vLLM.
- Réponses streaming et non-streaming avec mesure de latence et métriques.
- Gestion de tool calls : recherche web, accès fichier borné, actions Twitch et synthèse de réponse après outil.
- Mémoire RAG pour recherche de contexte court terme et long terme.
- Ingestion Twitch, OAuth, sondages, timeout/ban et modération.
- Points d'intégration STT/TTS pour interaction vocale.
- Contrôle VTube Studio pour expressions et mouvements.
- Panneau web pour les réglages runtime, la mémoire, la modération et les métriques.
- Scripts de génération de datasets, validation SFT/DPO, fine-tuning et quantification GPTQ.
- Sous-système chess/game-AI avec entraînement, évaluation et outils de match.

## Guide Des Dossiers

- `main.py` : coordinateur principal du runtime.
- `src/AI/LLM/` : client LLM, streaming, nettoyage JSON, schémas d'outils et exécution des outils.
- `src/AI/RAG/` : couche de mémoire et retrieval.
- `src/AI/TTS/` et `src/AI/STT/` : points d'intégration voix.
- `src/discord/` : pont Discord pour messages, vocal et lecture audio.
- `src/streaming/twitch/` : OAuth Twitch, chat, polls et actions de modération.
- `src/control_panel/` : aides backend pour le panneau web et la modération.
- `src/metrics/` : rendu de métriques de latence et d'utilisation tokens.
- `src/vtuber/` : contrôle des paramètres VTube Studio.
- `src/AI/game_ai/chess/` : outils d'entraînement, évaluation et inférence chess.
- `web/` : interface web de contrôle.
- `scripts/` : préparation de données, audits, fine-tuning, quantification, stress tests et migrations.

## Termes IA En Bref

- LLM : modèle de langage qui génère la réponse du personnage.
- RAG : recherche dans une mémoire avant de répondre.
- SFT : fine-tuning supervisé sur des exemples de bonnes réponses.
- DPO : entraînement par préférence avec de meilleures et de moins bonnes réponses.
- LoRA : adaptation légère d'un modèle sans tout ré-entraîner.
- GPTQ : quantification pour réduire la mémoire vidéo nécessaire.
- vLLM : serveur d'inférence rapide compatible avec l'API OpenAI.

## Confidentialité Et Éléments Retirés

Cette version exclut volontairement :

- fichiers `.env`, clés API, credentials Discord/Twitch/Google, tokens OAuth et whitelists ;
- fichiers de contexte/persona privés ;
- noms et identifiants de VIP/users ;
- poids de modèles, adaptateurs LoRA, checkpoints, tokenizers et exports quantifiés ;
- datasets SFT/DPO/TTS/chess, dumps Twitch, CSV/JSONL et données classées ;
- voix, audios générés, captures, logs, bases MLflow, caches et données runtime ;
- IP privées et ports locaux originaux.

Les noms ont été anonymisés : le projet s'appelle maintenant `AgathaAI`, le personnage `Agatha`, et les usernames en ligne ont été supprimés ou remplacés par des placeholders neutres.

## Comment Lire Le Projet

La meilleure lecture consiste à le voir comme une étude de cas d'ingénierie :

- commencer par `main.py` pour comprendre l'orchestration ;
- lire `src/AI/LLM/llm_agathaai_vision_vllm.py` et `src/AI/LLM/tools.py` pour le pipeline modèle/outils ;
- regarder `src/streaming/twitch/tw_plugin.py` et `src/control_panel/moderation.py` pour la modération Twitch ;
- examiner `scripts/fine_tune_qwen_vision.py`, `scripts/quant_qwen2.5-vl_to_gptq_int4.py` et les scripts de datasets pour la partie ML engineering ;
- ouvrir `web/index.html`, `web/frontend_scripts.js` et `web/backend_scripts.js` pour le panneau de contrôle.

## Compétences Démontrées

- conception de services Python asynchrones ;
- intégration multi-process entre Python, Node.js, UI web et serveurs de modèles locaux ;
- routing de prompts LLM, tool calls, streaming et nettoyage de sorties structurées ;
- modération, sécurité et limites d'accès fichier ;
- génération et validation de datasets, fine-tuning, évaluation et quantification ;
- métriques, feature flags, arrêt propre, synchronisation de configuration et gestion d'erreurs.
