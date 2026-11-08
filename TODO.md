# TODO: semestrálka 4IZ582

Stav k 8. 11. 2026. Zadání: https://4iz565.gitlab.io/project/

## Kontrola proti zadání

| Požadavek ze zadání | Stav | Kde v notebooku |
|---|---|---|
| Klasifikační úloha s nominálním targetem | ✅ | § 1 (`recommend`: v/x) |
| EDA: tvar a typy, chybějící hodnoty, poměr tříd, ≥ 2 smysluplné grafy | ✅ | § 3.1–3.3 (graf tříd + 3 panely) |
| Předzpracování v pandas, aspoň 2 operace | ✅ (všech 5) | § 2.1, § 4: mazání řádků, odvozené sloupce, groupby, chybějící hodnoty, join |
| Kódování a škálování v Pipeline / ColumnTransformer, ne před splitem | ✅ | § 7.1 |
| Testovací sada + stratifikovaná CV na trénovací části | ✅ | § 5.1 (`StratifiedGroupKFold`) |
| Aspoň 2 rodiny algoritmů | ✅ | § 7: LogisticRegression, HistGradientBoosting |
| Ladění přes Grid/RandomizedSearchCV s refit | ✅ | § 7.2 |
| Reportovat held-out (testový) odhad, ne `best_score_` | ✅ | § 8.2 (oboje vedle sebe, vysvětlené) |
| Accuracy + ROC AUC / AP | ✅ | § 8.2 |
| Confusion matrix při zvoleném prahu + zdůvodnění | ✅ | § 8.1, § 8.3 |
| ROC křivka | ✅ | § 8.4 |
| Metrika zdůvodněná poměrem tříd | ✅ | § 1.3, § 3.2 |
| Test použit jednou | ✅ | § 8.2 |
| Žádný leakage (preprocessing, target, temporal, duplicity / skupiny) | ✅ | § 1.2, § 5.1–5.4 |
| AI komponenta B: embeddingy jako vstup, srovnání s TF-IDF | ✅ | § 6, § 9 |
| Školní endpoint, klíč jen v env, cache výstupů, offline běh | ✅ | § 6.2 |
| Náklady AI komponenty (requesty, čas) | ✅ | § 9.5 |
| Restart & Run All bez chyb pod 30 min | ✅ lokálně (2,4 min) | ⚠️ ověřit na jupyter.vse.cz, bod 3 |
| Jen relativní cesty | ✅ | § 0 |
| Balíčky mimo standard s komentářem a install příkazem nahoře | ✅ | § 0 (`openai`) |
| Každé rozhodnutí vysvětlené v textu | ✅ | průběžně |
| **Kdo co dělal + co se dělalo s AI nástroji** | ❌ **chybí** | § 11, bod 1 a 2 |
| ZIP: .ipynb + HTML export + data CSV + cache | ✅ | `submission/4iz582_glassdoor_recommend.zip` |

Data, modelování, vyhodnocení i AI komponenta jsou kompletní. Před odevzdáním je potřeba doplnit sekci 11 a ověřit běh ve školním prostředí.

## Před odevzdáním notebooku (Ne 22. 11.)

- [ ] **1. Doplnit, kdo co dělal (§ 11).** Teď jsou tam jen šablony („to be completed by the team“). Domluvit se s Artemem a napsat konkrétně. Zadání chce, aby každý uvedl, kterou část napsal.
- [ ] **2. Napsat odstavec o použití AI nástrojů (§ 11).** Zadání vyžaduje odstavec o vlastním přínosu týmu. Musí být pravdivý a konkrétní. Podklad pro text:
  - Kostru notebooku, kód (načtení a vzorkování dat, pipeline, ladění, vyhodnocení, cache embeddingů), komentáře v notebooku, pomocné skripty v `dev/` a průvodce `docs/pruvodce.html` vygeneroval Claude Code (Anthropic).
  - Co jste udělali vy: zadání a volba datasetu, kontrola a pochopení kódu, rozhodnutí, která umíte obhájit (vyřazení hodnocení, split podle firem, pravidlo pro práh)…
  - Doplnit, co jste případně sami upravili.
  - Embedding model `qwen3-embedding:4b` je součást analýzy, ne pomůcka při psaní. To už v textu je.
- [ ] **3. Spustit na jupyter.vse.cz.** Nahrát obsah ZIPu a dát Restart Kernel and Run All Cells. Lokálně testováno jen na Pythonu 3.14 a nejnovějších knihovnách na 14 jádrech. Škola má Python 3.12 a možná starší pandas a scikit-learn a méně jader, takže ověřit, že projde bez chyb a pod 30 minut. Kdyby byl pomalý, snížit `n_iter` v § 7.2.
- [ ] **4. Ověřit limit velikosti souboru v INSIS.** ZIP má 58 MB, z toho 48 MB je cache embeddingů. Kdyby neprošel: zmenšit cache na 512 dimenzí (CV AUC 0,907 místo 0,911, cache zhruba poloviční) nebo dát cache jako odkaz ke stažení.
- [ ] **5. Po každé změně znovu vygenerovat výstupy.**
  ```bash
  .venv/bin/python dev/make_notebook.py      # jen pokud měníte generátor
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace term_project.ipynb
  .venv/bin/jupyter nbconvert --to html term_project.ipynb
  rm -f submission/*.zip && zip -r submission/4iz582_glassdoor_recommend.zip term_project.ipynb term_project.html requirements.txt data/glassdoor_sample.csv data/firms.csv data/embeddings_qwen3-embedding-4b.csv.gz data/embeddings_qwen3-embedding-4b.meta.json
  ```
  Rozhodněte se, jestli editujete generátor (`dev/make_notebook.py`), nebo přímo notebook. Obojí najednou se rozejde.
- [ ] **6. Projít notebook oba.** Ověřovací otázka na obhajobě míří na kód, data nebo rozhodnutí a odpovídá se bez spoluhráče. Podklad: `docs/pruvodce.html`, sekce „Otázky na obhajobu“.
- [ ] **7. Ověřit, že je tým a dataset zapsaný v INSIS** (termín byl 8. 10.).
- [ ] **8. Nahrát ZIP do INSIS.**

## Volitelná vylepšení (nejsou povinná)

- [ ] Bootstrap v § 9.1 losovat po firmách místo po recenzích (cluster bootstrap). Odstraní to omezení, které notebook sám přiznává.
- [ ] Explicitně projít Princeton reproducibility checklist (zadání ho uvádí jako referenci) a krátce shrnout v notebooku.
- [ ] Zadání mluví o cache „(CSV)“, naše je `.csv.gz` (komprimované CSV, pandas ho čte přímo). Případně zmínit v textu.
- [ ] Jazyk notebooku je angličtina (jako materiály kurzu). Přeložit do češtiny, pokud byste chtěli.
- [ ] Přidat Artema jako collaboratora do GitHub repa (Settings → Collaborators).

## Po odevzdání

- [ ] **Oponentní posudek, Ne 29. 11.**: jedno PDF za tým na notebook jiného týmu. Odpovědět na:
  1. Zvolili autoři vhodné algoritmy?
  2. Proběhlo ladění hyperparametrů tam, kde mělo?
  3. Jsou výsledky replikovatelné, je každý krok popsaný dost podrobně?
  4. Jsou zvolené správné metriky a správně interpretované?
  5. Je AI komponenta (nebo rozšíření) zdůvodněná a poctivě vyhodnocená?
  6. Jsou důležité kroky vysvětlené a zdůvodněné? Je text jasný a stručný?
- [ ] **Prezentace a obhajoba, 9. nebo 16. 12.** (rozpis bude 2. 12.). Povinná pro oba. Max. 8 minut, slidy doporučené. Otázky na dataset, předzpracování, modelování, AI komponentu, lessons learned a reakci na posudek. Připravit:
  - slidy (data → leakage rozhodnutí → split → TF-IDF vs. embeddingy → výsledky a learning curve → práh → lessons learned),
  - reakci na oponentní posudek, který dostanete,
  - každý zvlášť: umět vysvětlit libovolnou buňku notebooku.
